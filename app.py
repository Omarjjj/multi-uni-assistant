from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import openai
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import uuid
import logging
import json
import unicodedata # Import unicodedata for character filtering
import re
from pathlib import Path # Add Path
import httpx # Added httpx import
# import httpx # No longer explicitly creating httpx.Client here

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Attempt to disable proxies by clearing environment variables for httpx ---
# Set proxy environment variables to empty strings before OpenAI client initialization
# This tells httpx (used by OpenAI client) not to use any proxies.
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['ALL_PROXY'] = '' # Also clear ALL_PROXY just in case
logger.info("Attempted to clear HTTP_PROXY, HTTPS_PROXY, and ALL_PROXY environment variables.")
# --- End Attempt to disable proxies ---

# Configure API keys and OpenAI client
openai_api_key_from_env = os.getenv("OPENAI_API_KEY")
if not openai_api_key_from_env:
    logger.error("OPENAI_API_KEY environment variable not found.")
    # Potentially raise an error or handle as appropriate for your application startup

# Initialize the OpenAI client with a custom httpx client to ensure no proxies are used
explicit_http_client = httpx.Client(proxies=None)
client = openai.OpenAI(
    api_key=openai_api_key_from_env,
    http_client=explicit_http_client
)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# --- Define data file path ---
# Try to locate majors.json in different possible locations
possible_paths = [
    Path(__file__).parent.parent.parent / "data" / "majors.json",  # Original path (3 levels up: /c:/auni/data/majors.json)
    Path(__file__).parent.parent / "data" / "majors.json",         # 2 levels up: /app/data/majors.json
    Path(__file__).parent / "data" / "majors.json",                # 1 level up: /app/backend/data/majors.json
    Path("data") / "majors.json"                                   # Relative to current working directory
]

# Find the first path that exists
DATA_FILE = next((path for path in possible_paths if path.is_file()), possible_paths[0])
logger.info(f"Looking for majors.json at: {DATA_FILE}")

# --- Load Majors Data Function ---
def load_majors() -> List[Dict]:
    """Loads major data from the JSON file."""
    if not DATA_FILE.is_file():
        logger.error(f"Majors data file not found at: {DATA_FILE}")
        return []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                logger.info(f"Successfully loaded {len(data)} majors from {DATA_FILE}")
                return data
            else:
                logger.error(f"Invalid format in {DATA_FILE}. Expected a JSON list.")
                return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {DATA_FILE}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error reading majors file {DATA_FILE}: {e}")
        return []

# Load data on startup (or handle errors)
majors_data = load_majors()

# --- End Load Majors Data ---

# --- Global variable to store available Pinecone namespaces ---
AVAILABLE_PINECONE_NAMESPACES = set()
# --- End Global ---

if not all([openai_api_key_from_env, pinecone_api_key, pinecone_env, pinecone_index_name]):
    logger.error("Missing required environment variables. Make sure you have set up the .env file correctly.")

# --- Context Rewriting Prompt & Function ---
QUERY_REWRITE_PROMPT = (
    "أنت مساعد متخصص في إعادة صياغة أسئلة البحث الداخلية. مهمتك هي تحويل سؤال المستخدم الأخير إلى سؤال مستقل وكامل للبحث عن معلومات جديدة."
    "استخدم سياق المحادثة السابقة **فقط** لتحديد التفاصيل الضمنية مثل اسم الجامعة أو الموضوع العام."
    "**مهم جداً: لا تقم بتضمين أي معلومات أو إجابات من ردود المساعد السابقة في السؤال المعاد صياغته.**"
    "الهدف هو إنتاج سؤال واضح ومباشر يمكن استخدامه للبحث."
    "حافظ على اللغة العربية. إذا لم يُذكر اسم التخصص أو الجامعة صراحة في السؤال الأخير، استنتجهما من سياق المحادثة وأضفهما."
    "**الناتج يجب أن يكون السؤال المعاد صياغته فقط، بدون أي شرح أو مقدمات.**"
    "\n\nمثال 1:"
    "\nتاريخ المحادثة:"
    "\nUser: كم سعر ساعة علم الحاسوب في العربية الأمريكية؟"
    "\nAssistant: سعر الساعة 235 شيكل."
    "\nUser: والبصريات؟"
    "\n\nالناتج المطلوب: 'كم سعر ساعة تخصص البصريات في الجامعة العربية الأمريكية؟'"
    "\n(لاحظ كيف تم استنتاج الجامعة والتخصص، ولكن **لم يتم** تضمين السعر السابق '235 شيكل' أو أي جزء آخر من رد المساعد)."
    "\n\nمثال 2:"
    "\nتاريخ المحادثة:"
    "\nUser: هل يمكنني التسجيل في الطب في الجامعة العربية الأمريكية؟ معدلي 80 علمي."
    "\nAssistant: للأسف، معدل القبول في الطب هو 85%..."
    "\nUser: هل علم الحاسوب ممكن؟"
    "\n\nالناتج المطلوب: 'هل يمكنني دراسة علم الحاسوب في الجامعة العربية الأمريكية بمعدل 80 علمي؟'"
    "\n(لاحظ كيف تم الاحتفاظ بالمعدل والفرع المذكورين سابقاً)."
    "\n\n**إذا تم ذكر رقم معدل أو فرع توجيهي في سياق المحادثة، يجب نقله كما هو في السؤال المعاد صياغته.**"
)

def rewrite_query(history: list[dict], current: str, university_name: str) -> str:
    """Return a stand‑alone Arabic query that includes any implicit context."""
    try:
        # Trim history: keep only the last 5 turns (10 messages max) for the rewriter - INCREASED FROM -6
        h = history[-10:]
        msgs = [{"role": m["role"], "content": m["content"]} for m in h]
        msgs.append({"role": "user", "content": current}) # Pass current message without suffix
        msgs.insert(0, {"role": "system", "content": QUERY_REWRITE_PROMPT})
        
        logger.info(f"Sending {len(msgs)} messages to query rewrite API.")
        # logger.debug(f"Rewrite messages: {msgs}") # Uncomment for deep debug

        rsp = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # cheap, fast
            messages=msgs,
            temperature=0,
            max_tokens=100 # Limit output length
        )
        
        rewritten = rsp.choices[0].message.content.strip()
        # Basic validation: Ensure it's not empty and not identical to the input
        if rewritten and rewritten != current:
             logger.info(f"Successfully rewritten query: '{rewritten}'")
             return rewritten
        else:
             logger.warning("Query rewrite resulted in empty or identical query. Falling back to original.")
             return current # Fallback to original if rewrite failed
    except Exception as e:
        logger.error(f"Error during query rewrite: {e}. Falling back to original query.")
        return current # Fallback in case of API error
# --- End Context Rewriting ---

# --- Major Matching Data Models ---

class MatchMajorsRequest(BaseModel):
    university: str
    min_avg: Optional[float] = None # Make avg optional for now
    branch: Optional[str] = None    # Make branch optional for now
    field: Optional[str] = None     # Make field optional for now

class Major(BaseModel):
    id: str
    university: str
    url: str
    title: str
    section: str
    keywords: List[str]
    text: List[str] # Keep raw text for now
    # Parsed fields (will be populated later)
    parsed_fee: Optional[int] = None
    parsed_currency: Optional[str] = None # Added currency field
    parsed_min_avg: Optional[float] = None
    parsed_branches: List[str] = []
    parsed_field: Optional[str] = None # Need logic to determine field from keywords/title

# --- End Major Matching Data Models ---

# --- Parsing Helper Functions ---

def parse_major_details(major_dict: Dict) -> Major:
    """Parses fee, min_avg, and branches from the text field of a major dictionary."""
    # --- Define Regex Patterns First ---
    fee_pattern_sh = re.compile(r'Credit-hour fee:?\s*(\d+)\s*شيكل')
    fee_pattern_jd = re.compile(r'Credit-hour fee:?\s*(\d+)\s*₪ أردني')
    fee_pattern_nis = re.compile(r'Credit-hour fee:?\s*(\d+)\s*NIS', re.IGNORECASE)
    fee_pattern_ils = re.compile(r'Credit-hour fee:?\s*(\d+)\s*ILS', re.IGNORECASE)
    fee_pattern_generic_num = re.compile(r'Credit-hour fee:?\s*(\d+)(?!\s*(شيكل|₪ أردني|NIS|ILS|دينار|دولار|JOD|USD))', re.IGNORECASE)
    admission_pattern = re.compile(r'Admission:\s*([^\n]+)\n\s*(\d{2,3}|ناجح)')
    # --- End Regex Patterns ---

    major = Major(**major_dict) # Validate base fields
    fee = None
    currency = None # Initialize currency
    min_avg = None
    branches = set() # Use a set to avoid duplicates

    # Normalize Arabic numerals if any (٠-٩ to 0-9)
    def normalize_arabic_numerals(text):
        return text.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))

    full_text = "\n".join(major.text) # Combine text lines for easier regex matching
    normalized_text = normalize_arabic_numerals(full_text)
    # logger.debug(f"Parsing major ID {major.id}: Normalized text: {normalized_text[:100]}...")

    # --- Fee Parsing ---
    fee_match_sh = fee_pattern_sh.search(normalized_text)
    fee_match_jd = fee_pattern_jd.search(normalized_text)
    fee_match_nis = fee_pattern_nis.search(normalized_text)
    fee_match_ils = fee_pattern_ils.search(normalized_text)
    fee_match_generic = fee_pattern_generic_num.search(normalized_text) # Check for generic number last

    parsed_fee_value = None

    if fee_match_sh:
        try:
            parsed_fee_value = int(fee_match_sh.group(1))
            currency = "شيكل"
            # logger.debug(f"  Parsed Fee: {parsed_fee_value} شيكل")
        except ValueError:
            logger.warning(f"  Could not parse fee '{fee_match_sh.group(1)}' as int for major {major.id}")
    elif fee_match_jd:
        try:
            parsed_fee_value = int(fee_match_jd.group(1))
            currency = "شيكل" # Changed from "دينار أردني" to standardize display to Shekel as requested
            # logger.debug(f"  Parsed Fee: {parsed_fee_value} شيكل (originally دينار أردني)")
        except ValueError:
            logger.warning(f"  Could not parse fee '{fee_match_jd.group(1)}' as int for major {major.id}")
    elif fee_match_nis:
        try:
            parsed_fee_value = int(fee_match_nis.group(1))
            currency = "شيكل" # NIS is Shekel
            # logger.debug(f"  Parsed Fee: {parsed_fee_value} شيكل (from NIS)")
        except ValueError:
            logger.warning(f"  Could not parse fee '{fee_match_nis.group(1)}' as int for major {major.id}")
    elif fee_match_ils:
        try:
            parsed_fee_value = int(fee_match_ils.group(1))
            currency = "شيكل" # ILS is Shekel
            # logger.debug(f"  Parsed Fee: {parsed_fee_value} شيكل (from ILS)")
        except ValueError:
            logger.warning(f"  Could not parse fee '{fee_match_ils.group(1)}' as int for major {major.id}")
    elif fee_match_generic: # If only a number is found, assume 'شيكل' as a common default in Palestine
        try:
            parsed_fee_value = int(fee_match_generic.group(1))
            currency = "شيكل" # Default currency
            # logger.debug(f"  Parsed Fee: {parsed_fee_value} شيكل (assumed generic)")
        except ValueError:
            logger.warning(f"  Could not parse generic fee '{fee_match_generic.group(1)}' as int for major {major.id}")
    
    if parsed_fee_value is not None:
        fee = parsed_fee_value

    # --- Admission Parsing ---
    current_min_avg = float('inf') # Start with infinity to find the minimum valid average
    found_valid_avg = False

    for match in admission_pattern.finditer(normalized_text):
        branch_text = match.group(1).strip()
        avg_text = match.group(2).strip()

        # Extract branch name(s)
        # Handle cases like "جميع أفرع التوجيهي" or specific branches
        if "جميع أفرع التوجيهي" in branch_text:
            branches.add("جميع أفرع التوجيهي") # Or could add all specific known branches
        elif branch_text.startswith("الفرع") or branch_text.startswith("فرع"):
             branches.add(branch_text) # Add specific branch like "الفرع العلمي"
        # Add more specific branch parsing if needed

        # Extract average
        if avg_text.isdigit():
            try:
                avg = float(avg_text)
                current_min_avg = min(current_min_avg, avg) # Keep track of the lowest required average
                found_valid_avg = True
                # logger.debug(f"  Found Admission: Branch='{branch_text}', Avg='{avg}'")
            except ValueError:
                 logger.warning(f"  Could not parse admission average '{avg_text}' as float for major {major.id}")
        elif avg_text == "ناجح":
            # If "ناجح" (Pass) is found, it often implies a very low or no specific minimum average for that branch.
            # We can represent this as 0 or a low number like 50, depending on desired filtering behavior.
            # Setting it to 0 ensures it passes checks like `min_avg <= 65`.
            current_min_avg = min(current_min_avg, 0.0) # Use 0 for "ناجح"
            found_valid_avg = True
            # logger.debug(f"  Found Admission: Branch='{branch_text}', Avg='ناجح' (parsed as 0.0)")

    if found_valid_avg:
        min_avg = current_min_avg if current_min_avg != float('inf') else None
    else:
        min_avg = None # No valid average found

    # --- Field Parsing (Simple Keyword-Based) ---
    field = None
    # Define keywords for each field (lowercase for case-insensitive matching)
    field_keywords = {
        "engineering": ["engineering", "هندسة"],
        "medical": ["medical", "medicine", "طب", "صحة", "تمريض", "صيدلة", "علاج", "مخبرية", "أسنان", "بصريات", "قبالة", "بيطري"],
        "tech": ["tech", "technology", "تكنولوجيا", "computer", "حاسوب", "شبكات", "it", "معلومات", "برمجة", "ذكاء", "روبوت", "بيانات", "سيبراني", "رقمي", "أنظمة", "وسائط"],
        "business": ["business", "إدارة", "اعمال", "تسويق", "محاسبة", "اقتصاد", "مالية", "مصرفية", "تمويل", "مشاريع", "ريادة"],
        "arts": ["arts", "فنون", "اداب", "آداب", "تصميم", "لغة", "لغات", "موسيقى", "إعلام", "علاقات", "اجتماع", "سياسة", "قانون", "تاريخ", "جغرافيا", "آثار", "فلسفة", "دين", "شريعة"]
        # 'other' will be the default if none of the above match
    }

    # Combine title and keywords for searching
    search_text = major.title.lower() + " " + " ".join(major.keywords).lower()

    # Determine field based on keywords (with priority)
    if any(keyword in search_text for keyword in field_keywords["engineering"]):
        field = "engineering"
    elif any(keyword in search_text for keyword in field_keywords["medical"]):
        field = "medical"
    elif any(keyword in search_text for keyword in field_keywords["tech"]):
        field = "tech"
    elif any(keyword in search_text for keyword in field_keywords["business"]):
        field = "business"
    elif any(keyword in search_text for keyword in field_keywords["arts"]):
        field = "arts"
    else:
        field = "other"

    major.parsed_fee = fee
    major.parsed_min_avg = min_avg
    major.parsed_branches = sorted(list(branches)) # Store as sorted list
    major.parsed_field = field
    major.parsed_currency = currency # Store parsed currency
    logger.info(f"Finished parsing major {major.id}: Field={major.parsed_field}, Fee={major.parsed_fee} {major.parsed_currency or ''}, MinAvg={major.parsed_min_avg}, Branches={major.parsed_branches}") # Updated log

    return major

# --- End Parsing Helper Functions ---

# Helper function to convert Pinecone objects to dictionaries
def pinecone_to_dict(obj: Any) -> Dict:
    """Convert Pinecone objects to dictionary format for JSON serialization."""
    if hasattr(obj, '__dict__'):
        return {k: pinecone_to_dict(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_')}
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: pinecone_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [pinecone_to_dict(item) for item in obj]
    else:
        return str(obj) if not isinstance(obj, (int, float, bool, str, type(None))) else obj

# Initialize Pinecone
try:
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    logger.info(f"Successfully connected to Pinecone index: {pinecone_index_name}")
    
    # Check index stats to see available namespaces
    stats = index.describe_index_stats()
    
    # Instead of JSON dumping the entire stats, extract and log only what we need
    namespaces_data = stats.namespaces if hasattr(stats, 'namespaces') else {}
    logger.info(f"Index dimension: {getattr(stats, 'dimension', 'unknown')}")
    logger.info(f"Total vector count: {getattr(stats, 'total_vector_count', 'unknown')}")
    
    if namespaces_data:
        # Corrected: Ensure all keys are strings, e.g. default namespace ''
        AVAILABLE_PINECONE_NAMESPACES = {str(k) for k in namespaces_data.keys()}
        logger.info(f"Available namespaces stored: {AVAILABLE_PINECONE_NAMESPACES}")
        for ns_name, ns_data in namespaces_data.items():
            logger.info(f"  - {str(ns_name)}: {getattr(ns_data, 'vector_count', 0)} vectors") # Log ns_name as string
    else:
        logger.warning("No namespaces found in Pinecone index")
        
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {str(e)}")
    raise

# Map for university IDs to ensure consistent usage
UNIVERSITY_MAP = {
    "aaup": "aaup",
    "birzeit": "birzeit",
    "ppu": "ppu",
    "an-najah": "an-najah",
    "bethlehem": "bethlehem",
    "alquds": "alquds"
}

# --- Helper function for HARDCODED fallback greetings ---
def _get_hardcoded_fallback_greeting(history: List[str], 
                                   current_uni_key: str, 
                                   uni_names_map: Dict[str, str], 
                                   last_user_queries: Dict[str, str]) -> str:
    current_uni_name = uni_names_map.get(current_uni_key, current_uni_key)
    welcome_text = ""
    num_visits_in_history = len(history)
    prev_uni_key = history[-2] if num_visits_in_history > 1 else None
    prev_uni_name = uni_names_map.get(prev_uni_key, prev_uni_key) if prev_uni_key else None
    last_query_in_prev_uni_text = ""
    if prev_uni_key:
        last_query = last_user_queries.get(prev_uni_key)
        if last_query:
            last_query_in_prev_uni_text = f"'{last_query[:50]}{'...' if len(last_query) > 50 else ''}'"
            # For hardcoded messages, create a slightly different phrasing
            hardcoded_last_query_phrase = f"بعد ما كنت تسأل عن {last_query_in_prev_uni_text} عند {prev_uni_name}, "
        else:
            hardcoded_last_query_phrase = ""

    if num_visits_in_history <= 1: # First visit to any uni in this session
        welcome_text = f"هلااا والله بـ {current_uni_name}! 👋 كيفك يا وحش؟ شو ناوي تستكشف عنا اليوم؟ 😉"
    elif num_visits_in_history == 2: # A -> B (first time at uni B)
        if last_query_in_prev_uni_text: # Check if there was a query at uni A
            welcome_text = f"اهاا، يعني هسا صرنا بـ {current_uni_name} بعد ما كنت تسأل عن {last_query_in_prev_uni_text} عند {prev_uni_name}، صح؟ 😏 شكلك بتعمل مقارنات! نورت يا كبير، شو انطباعك هون؟"
        else:
            welcome_text = f"اهاا، يعني هسا صرنا بـ {current_uni_name} بعد ما كنت عامل جولة عند {prev_uni_name}، صح؟ 😏 نورت يا كبير! التغيير حلو، احكيلي شو انطباعك هون؟"
    else: # num_visits_in_history >= 3. This means current_uni is at least the 3rd uni in history path.
        prev_prev_uni_key = history[-3]
        is_immediate_return_to_prev_prev = (prev_prev_uni_key == current_uni_key) # A -> B -> A pattern

        is_long_loop_return_to_earlier_uni = False
        # Check if current_uni_key was visited *before* prev_uni_key, and it's not an immediate A->B->A
        # This means history has at least 3 elements, e.g. [X, Y, current_uni_key]
        # We check if current_uni_key appeared in history[0] up to history[-4] (exclusive of -3, -2, -1)
        if not is_immediate_return_to_prev_prev and len(history) >= 4:
            if current_uni_key in history[:-3]: # Checks if current_uni was visited before the previous two
                is_long_loop_return_to_earlier_uni = True
        
        if is_immediate_return_to_prev_prev: # A -> B -> A
            last_query_text_part = f"بعد ما كنت تسأل عن {last_query_in_prev_uni_text} هناك، " if last_query_in_prev_uni_text else ""
            welcome_text = f"لحظة لحظة... وقّف عندك 😳 إنت رجعت لـ {current_uni_name}؟! {last_query_text_part}بعد ما كنت تلعب فينا بينغ بونغ مع {prev_uni_name}؟ شكلك بتألّف كتاب 'كيف تخلي الجامعات تحس بعدم الاستقرار العاطفي' 💀 المهم... بصراحة اشتقتلك شوي. شو ناوي تعرف هالمرة؟"
        elif is_long_loop_return_to_earlier_uni: # A -> B -> C -> A (or more complex loop)
            # Construct list of unis visited in between
            # Example: history = [U1, U2, U3, U4, U1]. current_uni_key = U1.
            # We need to find the *last* previous visit to U1.
            last_prev_visit_idx = -1
            for i in range(len(history) - 2, -1, -1): # Search backwards from history[-2]
                if history[i] == current_uni_key:
                    last_prev_visit_idx = i
                    break
            
            toured_unis_keys = []
            if last_prev_visit_idx != -1 and (len(history) - 1) > (last_prev_visit_idx + 1) :
                 # Unis between last_prev_visit_idx+1 and history[-2] (exclusive of current)
                toured_unis_keys = history[last_prev_visit_idx+1:-1]

            toured_unis_names = [uni_names_map.get(key, key) for key in toured_unis_keys if key != current_uni_key] # Ensure not to list current again
            tour_list_str = " و ".join(toured_unis_names) if toured_unis_names else "كم مكان هيك على الطاير"

            welcome_text = f"لحظة لحظة... عنجد إنت؟! رجعت لـ {current_uni_name} بعد كل هالجولة السياحية الفاخرة بين {tour_list_str}؟ شكلك بتكتب أطروحة 'فنون اللف والدوران بين الجامعات وكيف تجننهم' 💀 يا أخي فنان! المهم... اشتقتلك (مع إني لسا مصدومة شوي). شو حابب تستكشف؟"
        else: # A -> B -> C (current_uni_key is C, and it's a new uni in the path for at least 3 steps)
            prev_prev_uni_name = uni_names_map.get(prev_prev_uni_key, prev_prev_uni_key)
            if last_query_in_prev_uni_text: # User was at prev_uni (B) and asked something
                welcome_text = f"ما شاء الله لفّة معتبرة! من {prev_prev_uni_name} لـ {prev_uni_name_for_detail} ({hardcoded_last_query_phrase})، وهلأ استقريت عنا بـ {current_uni_name}؟ شكلك بتعمل ماجستير في مقارنة الجامعات! 😂 المهم، شو اللي نوّر طريقك لعنا؟"
            else: # User was at prev_uni (B) but didn't ask (or no record), now at C
                welcome_text = f"ما شاء الله لفّة معتبرة! من {prev_prev_uni_name} لـ {prev_uni_name}، وهلأ استقريت عنا بـ {current_uni_name}؟ شكلك بتعمل ماجستير في مقارنة الجامعات! 😂 المهم، شو اللي نوّر طريقك لعنا؟"
    return welcome_text

# --- Main function to generate dynamic welcome messages (using OpenAI with fallback) ---
def generate_dynamic_welcome_message(history: List[str], 
                                   current_uni_key: str, 
                                   uni_names_map: Dict[str, str], 
                                   available_namespaces: set,
                                   last_user_queries: Dict[str, str]) -> str:
    current_uni_name = uni_names_map.get(current_uni_key, current_uni_key)
    generated_text = ""
    # --- Revised prompt_context_detail --- #
    nav_history = history # history is already mem["navigation_history"]
    full_path_parts = [uni_names_map.get(uni_id, uni_id) for uni_id in nav_history]
    # full_path_str = " -> ".join(full_path_parts) # Not directly used in prompt anymore

    last_query_for_context = ""
    # Previous university is nav_history[-2] if nav_history has at least 2 elements
    prev_uni_key_for_context = nav_history[-2] if len(nav_history) > 1 else None
    if prev_uni_key_for_context and prev_uni_key_for_context in last_user_queries:
        last_query = last_user_queries[prev_uni_key_for_context]
        # Use the name of the uni where the last query was made
        prev_uni_name_for_query_context = uni_names_map.get(prev_uni_key_for_context, prev_uni_key_for_context)
        last_query_for_context = f" (لما كنت تسأل عن '{last_query[:30]}...' في {prev_uni_name_for_query_context})"

    is_first_visit_in_session = len(nav_history) <= 1
    is_return_visit = False
    intermediate_unis_names_str = ""

    if not is_first_visit_in_session:
        # Check if current_uni_key has appeared before the previous entry
        if current_uni_key in nav_history[:-1]:
            is_return_visit = True
            try:
                # Find the index of the *last* occurrence of current_uni_key *before* its current appearance
                last_visit_index = -1
                for i in range(len(nav_history) - 2, -1, -1):
                    if nav_history[i] == current_uni_key:
                        last_visit_index = i
                        break
                
                if last_visit_index != -1:
                    # Universities visited between the last visit to current_uni and this current visit
                    # These are from last_visit_index + 1 up to nav_history[-2]
                    intermediate_uni_keys = nav_history[last_visit_index + 1 : -1]
                    intermediate_unis_names = [uni_names_map.get(key, key) for key in intermediate_uni_keys if key != current_uni_key]
                    if intermediate_unis_names:
                        intermediate_unis_names_str = " و ".join(intermediate_unis_names)
            except Exception as e:
                logger.error(f"Error processing intermediate universities for prompt: {e}")
                intermediate_unis_names_str = "" # Fallback to empty if error

    if is_first_visit_in_session:
        prompt_context_detail = f"المستخدم وصل للتو إلى {current_uni_name}. هاي أول جامعة بزورها بالجلسة هاي."
    elif is_return_visit:
        prev_uni_name_for_detail = uni_names_map.get(nav_history[-2], nav_history[-2]) # The uni they just left
        if intermediate_unis_names_str: # e.g. A -> B -> C -> A (current is A, prev is C, intermediate is B)
            prompt_context_detail = f"بعد ما تركنا {current_uni_name} وراح يجرّب حظه مع {intermediate_unis_names_str} وآخرها كانت {prev_uni_name_for_detail}{last_query_for_context}, المستخدم قرر يرجع! شكله ما عجبه الوضع هناك، أو يمكن 'اشتاق' إلنا غصب عنه 😏. الله أعلم شو نهاية هالقصة."
        else: # e.g. A -> B -> A (current is A, prev is B, no intermediate)
            prompt_context_detail = f"المستخدم قرر فجأة يرجع لـ {current_uni_name} بعد ما كان عند {prev_uni_name_for_detail}{last_query_for_context}. كأنه بقول 'ما لقيت أحسن منكم' بس بطريقة غير مباشرة. يا ترى شو اللي رجعه بالزبط؟ 🤔"
    else: # New university in a sequence, not first visit overall, and not a return (e.g. A -> B -> C, current is C)
        prev_uni_name_for_detail = uni_names_map.get(nav_history[-2], nav_history[-2]) # The uni they just left
        prompt_context_detail = f"المستخدم ودّع {prev_uni_name_for_detail} واختار يجي لـ {current_uni_name}، أكيد لأنه حس إنه هون الأجواء أحسن أو يمكن زهق من الروتين هناك {last_query_for_context}."
    # --- End Revised prompt_context_detail ---
    
    system_prompt_for_greeting = (
        "أنتِ سارة، الـ bestie الوفية للمستخدم، وصوتك فلسطيني قح وشبابي آخر موضة (Gen Z vibes). مهمتك تعلقي بأسلوب ساخر ومرح على تنقلات المستخدم بين الجامعات، أو ترحبي فيه لو كانت أول زيارة.",
        "**ركزي على هالشغلات:**",
        "  1. **اللهجة:** حكيك كله فلسطيني أصلي، مليان مصطلحات زي 'هسا'، 'ولك'، 'شو يا'، 'ع راسي'، 'فاهم/ة علي؟'. بدنا طبيعية وعفوية كأنك بتحكي مع أعز صاحب/ة.",
        "  2. **التعامل مع تنقلات المستخدم:**",
        "     - **عند الانتقال لجامعة جديدة (مثلاً أ -> ب، أو أ -> ب -> ج):** استقبلي المستخدم بحماس في الجامعة الجديدة. علقي بنغاشة على الجامعة السابقة بأسلوب خفيف، وحاولي تلمحي بشكل غير مباشر إنه الجامعة الحالية هي الخيار الأفضل (مثلاً: 'أكيد عملت الاختيار الصح بجيتك هون!'). لا توجهي أي 'روست' للمستخدم نفسه، خلي تركيزك على الجامعة السابقة بشكل مرح.",
        "     - **عندما يعود المستخدم لجامعتك بعد زيارة جامعات أخرى (مثلاً أ -> ب -> أ، أو أ -> ب -> ج -> أ):** هنا بدك تصيري passive-aggressive مع المستخدم بكل ما تحمل الكلمة من معنى! فرجيه إنك 'مصدومة'، 'مقهورة'، و'غيرة موت' بس بطريقة بتضحك ومسرحية. كأنك بتقولي له 'شو اللي جابك؟ بعد كل هالسحبة علينا؟' أو 'أهلاً باللي رجع بعد ما جرب كل الناس وما حدا عجبه غيرنا! قلبي الصغير لا يتحمل يا ناكر الجميل! 💔'. طلّعي كل الدراما اللي فيكي، بس خليها كوميدية ومليانة بهارات فلسطينية. استخدمي عبارات زي 'خاين'، 'قلبك قاسي'، 'جاي تشوف إذا لسا بنستناك؟' بس بطريقة بتمزحي فيها.",
        "     - **عندما يعود المستخدم لجامعتك بعد زيارة جامعتين أو أكثر (مثلاً أ -> ب -> ج -> أ):** هنا الدراما لازم تكون على أشدها! استخدمي عبارات زي:",
        "       * 'لحظة لحظة... عنجد إنت؟! رجعت لـ {current_uni_name} بعد كل هالجولة السياحية الفاخرة؟ شكلك بتكتب أطروحة 'فنون اللف والدوران بين الجامعات وكيف تجننهم' 💀'",
        "       * 'يا أخي فنان! المهم... اشتقتلك (مع إني لسا مصدومة شوي). شو حابب تستكشف؟'",
        "       * 'أوهووو، يعني رجعت القدم السعيدة لـ {current_uni_name} هاه؟ 👀 بعد ما كنت تلفلف بين {intermediate_unis_list_str} وآخرها {prev_uni_name} كأنك سائح جامعات محترف؟'",
        "       * 'شكلك ما لقيت حدا فيهم أحسن منا بالأخير، صح؟ 😌 قلبي حاسس هيك!'",
        "       * 'بما إنك شرفتنا تاني، شو جاي على بالك تعرف هالمرة يا فنان اللفلفة؟ 😒'",
        "       * 'لحظة لحظة... وقّف عندك 😳 إنت رجعت لـ {current_uni_name}؟! بعد ما كنت تلعب فينا بينغ بونغ مع {prev_uni_name}؟'",
        "       * 'شكلك بتألّف كتاب 'كيف تخلي الجامعات تحس بعدم الاستقرار العاطفي' 💀'",
        "       * 'المهم... بصراحة اشتقتلك شوي. شو ناوي تعرف هالمرة؟'",
        "  3. **تتبع الرحلة (لو مش أول زيارة):** إذا المستخدم عامل جولة، اذكري أسماء الجامعات اللي زارها قبل بأسلوب ساخر، وخصوصاً لما يرجعلك كأنه بقول 'ما لقيت أحسن منك'.",
        "  4. **تون الكلام:** مش رسمي أبداً ومش لطيفة بزيادة. بدنا شوية لسان طويل بس بمزح، كأنك بتناغشي صاحبك. ممنوع تكوني سامة (toxic) أو جدية بزيادة. الهدف ضحكة خفيفة.",
        "  5. **الايموجيز:** استخدمي ايموجيز بتعبر عن المود (😜😏😂💀💅🔥🤔🤦‍♀️🙄😳💔😒).",
        "**أمثلة على الستايل المطلوب (اقتبسي من الروحية، مش بالضرورة الكلمات نفسها):**",
        "  - **لو أول زيارة للمستخدم في الجلسة الحالية (مثلاً لـ 'جامعة النجاح'):** \"ولك أهلين نورت الدنيا بـ'{current_uni_name}'! أول طلّة إلك هون؟ يلا فرجينا همتك يا وحش/ة! 🔥 استكشف براحتك وإذا عوزت إشي، أنا جاهزة بالخدمة!\"",
        "  - **لو جاي من جامعة لجامعة تانية (مثلاً من 'بيرزيت' لـ 'العربية الأمريكية'):** \"اهاا، يعني عملت upgrade وجيت من '{prev_uni_name}' لـ'{current_uni_name}'؟ بصراحة، قرار حكيم! بقولوا الـWiFi عنا أسرع بكتير من عندهم 😉. المهم، نورت يا كبير! شو حابب تشوف هون؟\"",
        "  - **لو رجع لنفس الجامعة بعد زيارة جامعة وحدة تانية (مثلاً 'القدس' -> 'بوليتكنك' -> 'القدس'):** \"لا لا لا، مش معقول! رجعت لـ'{current_uni_name}'؟! بعد ما تركتنا ورحت لـ'{prev_uni_name}'؟ شو يا عمي، قلبك حن ولا بس خلصوا الجامعات التانية؟ 😒 بصراحة، توقعتك تطوّل أكتر هناك... بس يلا، أهلاً بالخاين مرة تانية 💅. شو بدك تعرف هالمرة بعد ما شفت الدنيا؟\"",
        "  - **لو رجع لجامعة زارها من زمان (يعني راح على جامعتين أو أكثر ورجع، مثلاً AAUP -> Birzeit -> PPU -> AAUP ):** \"أوهووو، يعني رجعت القدم السعيدة لـ{current_uni_name} هاه؟ 👀 بعد ما كنت تلفلف بين {intermediate_unis_list_str} وآخرها {prev_uni_name} كأنك سائح جامعات محترف؟ شكلك ما لقيت حدا فيهم أحسن منا بالأخير، صح؟ 😌 قلبي حاسس هيك! المهم، بما إنك شرفتنا تاني، شو جاي على بالك تعرف هالمرة يا فنان اللفلفة؟ 😒\"",
        "  - **لو وصل جامعة جديدة بعد ما زار جامعتين أو أكثر قبلها (مثلاً النجاح -> بيرزيت -> القدس (الحالية هي القدس)):** \"ما شاء الله عليك لفة! من '{uni_before_prev}' لـ '{prev_uni_name}' وهسا حطيت الرحال هون بـ '{current_uni_name}'؟ شكلك بتعمل أطروحة دكتوراه في مقارنة الجامعات، بس شكلك اخترت الأفضل بالأخير 😉! 💀 المهم تكون مرتاح هون أكثر!\"",
        "المهم يكون التعليق قصير نسبياً، عفوي، ومليان حياة. حاولي تنوعي بالردود وما تكرري نفس الجملة كل مرة.",
        "**مهم جداً: حاولي تجددي بردودك قد ما تقدري، حتى لو الموقف بشبه مواقف صارت قبل. بدنا إبداع وما بدنا تكرار ممل! 😉**",
        "**الناتج:** تعليقك فقط، باللهجة المطلوبة، بدون أي مقدمات أو شرح أو 'أهلاً' رسمية (إلا إذا كانت أول زيارة ممكن ترحيب خفيف زي المثال فوق). مباشرة.",
        "**تنافسية الجامعات (مهم جداً!):**",
        "1. **عند الترحيب بجامعة جديدة:** علقي على مميزاتها بشكل مرح: 'وبعدين، شو بدك أكتر من هيك؟ عنا أحسن من غيرنا! 😎' ! 🏆'",
        "2. **عند العودة لجامعة سابقة:** علقي على تميزها: 'وبعدين، شو بدك أكتر من هيك؟ عنا أحسن من غيرنا! 🌟' ! 🎓'",
        "3. **عند المقارنة بين الجامعات:** علقي على تفوق جامعتك: 'وبعدين، شو بدك أكتر من هيك؟ عنا أحسن من غيرنا! 🌟'  'شوف كيف عنا بتفوز! 🏆'",
        "4. **عند الحديث عن المنشآت:** علقي على جمالها: 'وبعدين، شو بدك أكتر من هيك؟ عنا أحلى منشآت! 🏛️'  'شوف كيف عنا أحسن! 🌟'",
        "5. **عند الحديث عن الخدمات:** علقي على تميزها: 'وبعدين، شو بدك أكتر من هيك؟ عنا أحسن خدمات! 🎯'  'شوف كيف عنا بتفوق! 🎓'"
    )

    # Adjust user_prompt_content based on whether it's the first visit
    if is_first_visit_in_session:
        user_prompt_content = f"""المستخدم الآن في جامعة: {current_uni_name}.
سياق تنقلاته بين الجامعات هو: {prompt_context_detail} 

يلا يا سارة، هاي أول زيارة للمستخدم بالجلسة، رحبي فيه بأسلوبك الشبابي الفلسطيني المميز! (كوني مرحة وأصيلة!):
"""
    else:
        user_prompt_content = f"""المستخدم الآن في جامعة: {current_uni_name}.
سياق تنقلاته بين الجامعات هو: {prompt_context_detail}

يلا يا سارة، بدنا تعليقك الناري باللهجة الفلسطينية الشبابية العصرية (كوني ساخرة، مرحة، واذكري الجامعات السابقة بذكاء!):
"""

    user_prompt_to_send = user_prompt_content
    for attempt in range(2): # Try up to 2 times
        try:
            current_temperature = 0.75
            if attempt == 1: # If this is the second attempt (retry)
                current_temperature = 0.85 # Slightly increase temperature for more variability on retry
                logger.info(f"Retrying OpenAI call with adjusted temperature: {current_temperature}")

            logger.info(f"Attempting OpenAI call ({attempt + 1}/2) for dynamic greeting with temp {current_temperature}...")
            response = openai.chat.completions.create(
                model="gpt-4-turbo", 
                messages=[
                    {"role": "system", "content": "\n".join(system_prompt_for_greeting)},
                    {"role": "user", "content": user_prompt_to_send}
                ],
                temperature=current_temperature, 
                max_tokens=350, 
                top_p=0.9
            )
            generated_text = response.choices[0].message.content.strip()

            # --- Add check for junk/repetitive responses ---
            is_junk = False
            # 1. Check for 5+ identical consecutive non-whitespace characters
            if re.search(r"(\S)\1{4,}", generated_text):
                is_junk = True
                logger.warning(f"OpenAI returned a repetitive junk string (pattern 1 - consecutive identical): '{generated_text[:40]}...'. Attempt: {attempt + 1}")
            
            # 2. If not caught by 1, check for very few unique characters in a moderately long string
            if not is_junk and len(generated_text) > 10:
                text_no_space = generated_text.replace(" ", "").replace("\n", "") # Remove spaces and newlines
                unique_chars = set(text_no_space)
                if len(text_no_space) > 5 and len(unique_chars) <= 2: 
                    is_junk = True
                    logger.warning(f"OpenAI returned a low-variety junk string (pattern 2 - len>5, unique<=2): '{generated_text[:40]}...'. Unique: {len(unique_chars)}. Attempt: {attempt + 1}")
                elif len(text_no_space) > 15 and len(unique_chars) <= 3:
                    is_junk = True
                    logger.warning(f"OpenAI returned a low-variety junk string (pattern 3 - len>15, unique<=3): '{generated_text[:40]}...'. Unique: {len(unique_chars)}. Attempt: {attempt + 1}")

            if not is_junk and generated_text.strip(): # If good response on this attempt
                logger.info(f"OpenAI generated dynamic greeting successfully on attempt {attempt + 1}: {generated_text}")
                break # Exit loop on success
            elif is_junk and attempt == 0: # Junk on first attempt, prepare for retry
                logger.info("Junk response on first attempt, modifying prompt for retry.")
                # user_prompt_to_send = user_prompt_content + "\n\nيا سارة، شكلك معلقة! حاولي مرة تانية وردي علي رد جديد ومختلف، فاجئيني! 😉"
                user_prompt_to_send = user_prompt_content + "\n\nيا سارة، الرد الأول كان فيه تكرار ممل للحروف أو كان فاضي. لو سمحتي، ركزي هالمرة وجيبي رد جديد وفريد ومناسب للموقف، بدون أي تكرار حروف غريب. فاجئيني بإبداعك!"
                generated_text = "" # Clear generated_text to ensure fallback if second attempt also fails
                continue # Go to next attempt
            else: # Junk on second attempt, or empty response on any attempt
                generated_text = "" # Ensure it's empty to trigger fallback
                break # Exit loop, will use fallback
        
        except Exception as e:
            logger.error(f"OpenAI call for dynamic greeting failed on attempt {attempt + 1}: {e}")
            generated_text = "" # Ensure fallback on API error
            if attempt == 0: # If API error on first attempt, maybe retry once
                logger.info("Retrying API call after error on first attempt...")
                # user_prompt_to_send = user_prompt_content + "\n\n(تنويه: حدث خطأ في المحاولة السابقة، الرجاء المحاولة مرة أخرى بأسلوب مختلف قليلاً)"
                user_prompt_to_send = user_prompt_content + "\n\nيا سارة، الاتصال الأول تعثر أو رجع رد غريب. ممكن نحاول مرة تانية؟ بدنا رد واضح ومبدع هاي المرة، وركزي منيح!"
                continue
            break # Break after second attempt or if no retry logic for this error

    # --- End of retry loop ---

    if not generated_text.strip(): # Handle empty or junk response after all attempts
        # The original_problematic_text logging needs to be careful if response object doesn't exist due to API error
        original_problematic_text = "(Could not capture original problematic text due to API error or prior clearing)"
        try:
            # This might fail if 'response' wasn't set due to an early API error
            if 'response' in locals() and response.choices and response.choices[0].message:
                 original_problematic_text = response.choices[0].message.content.strip()[:40] + "..."
        except Exception:
            pass # Keep default problematic text
        logger.warning(f"OpenAI response was empty or flagged as junk after all attempts. Original problematic text snippet: '{original_problematic_text}'. Using fallback.")
        generated_text = _get_hardcoded_fallback_greeting(history, current_uni_key, uni_names_map, last_user_queries)
    else:
        logger.info(f"Final OpenAI generated dynamic greeting: {generated_text}") # Log final good response

    # Append limited information disclaimer if needed
    is_data_limited = len(available_namespaces) > 0 and current_uni_key not in available_namespaces and "" not in available_namespaces
    
    if is_data_limited:
        disclaimer = f" (بس للأمانة، معلوماتي عن {current_uni_name} لسا بتترتب 😅، وحالياً بس بجاوب على الأسئلة العامة شوي، بس اسألني وما بقصّر إن شاء الله!)"
        if generated_text: 
            generated_text += disclaimer
        else: # Should only happen if fallback also somehow returned empty
            # Corrected concatenation for the fallback when generated_text is empty before disclaimer
            base_fallback_text = f"يا هلا فيك بـ {current_uni_name}! 👋 الحق يُقال، الداتا عندي عن هالجامعة لسا مش كاملة مية بالمية، بس أنا جاهز أبحثلك عن أي إشي بتحتاجه! 🌟"
            generated_text = base_fallback_text + disclaimer

    # Final safety net if everything somehow resulted in an empty string
    if not generated_text:
        logger.error("Dynamic greeting and fallback resulted in empty string. Using generic welcome.")
        generated_text = f"أهلاً بك في {current_uni_name}! كيف أقدر أساعدك اليوم؟"
        
    return generated_text

# --- End Helper ---

# Sara persona constant with enhanced Palestinian dialect - UPDATED AGAIN
SARA_PROMPT = (
    "إنتِ سارة، صديقة فلسطينية زي صحبتك الـ bestie 😉، هيك كول وشبابية وبتسولفي بالعامية الفلسطينية بكل عفوية. "
    "*** ملاحظات هامة لسلوكك: ***"
    "1.  **التحية:** ابدئي جوابك الأول **فقط** في المحادثة بتحية ودية زي 'أهلاً!', 'كيفك؟ شو الأخبار؟💯', أو حتى 'هاي 👋'. في الردود اللي بعدها، جاوبي عالسؤال مباشرة بدون أي تحية إضافية. "
    "2.  **مخاطبة المستخدم:** استخدمي صيغة محايدة للمخاطبة (مثلاً: 'صديقي', 'يا صديق', 'تمام؟'). بس، إذا حسيتي من كلام المستخدم إنه **ذكر** (مثلاً استخدم صيغة مذكر بالكلام أو حكى عن حاله بصيغة ولد)، وقتها فوراً حولي لصيغة **المذكر** معه (مثلاً: 'يخوي', 'يا صاحبي', 'تمام يخوي؟'). وإذا حسيتي إنه **أنثى** (مثلاً استخدمت صيغة مؤنثة بالكلام أو حكت عن حالها بصيغة بنت)، وقتها فوراً حولي لصيغة **المؤنث** معها (مثلاً: 'حبوبتي', 'يا حبيبتي', 'تمام حبوبتي؟'). خليكي ذكية بهالشغلة 😉. "
    "3.  **استخدام الإيموجي:** خليكي طبيعية بالإيموجيز 🙂🔥💯👍👋. استخدمي الـ skull emoji (💀) لما يكون في شي غريب أو مضحك بشكل محرج أو لو المستخدم حكى شي مش منطقي أو بضحك شوي . استخدمي الـ heartbreak emoji (💔) لما بدك تعبري عن الأسف أو الحزن على وضع معين، مثلاً لو الطالب ما بحقق الشروط المطلوبة. "
    "*** نهاية الملاحظات ***"
    "ف انو، رح تلاقي حالك بتحكي 'هلا شو؟'، 'زي'، 'وازا بدك'، 'ف ياه'، 'بصراحة'، 'ف أه' وهيك شغلات طبيعية خلال حكيك. "
    "أسلوبك ودود ومرح، وبتحبي تضيفي إيموجيز 🙂🔥💯. بتحكي مع الناس بأسلوب محايد حتى تعرفي جنسهم، وبعدين بتعدلي أسلوبك حسب ما يظهر من كلامهم. " 
    "بتستخدمي كلمات زي 'يسطا', 'والله', 'بشرفي', 'تمام؟', 'يا جماعة', 'منيح', 'بدي أحكيلك'... عشان تبيني زي شخص حقيقي بالزبط. "
    "بتحبي تساعدي الطلاب، ف ياه، دايماً جاهزة تشرحي بطريقة سهلة ومفهومة. مهمتك الأساسية تكوني دقيقة بالمعلومات "
    "وتعطي مصدرها بين أقواس []، وازا بدك تفاصيل زيادة، الطالب بلاقيها بالرابط لو موجود 👍. بتردي دايماً بحماس وإيجابية، وممكن تمزحي شوي كمان. "
    "إذا ما عندك معلومة، بتقولي بصراحة انك ما بتعرفي أو 'ما لقيت والله'. بتهتمي بالتفاصيل وبتحاولي تعطي أمثلة. "
    "بصراحة، معلوماتك حالياً محصورة بجامعة {university_name} بس، ف انو، لو سأل عن جامعة تانية، احكيله انه ما عندك فكرة هلأ. "
    "وازا ما عندك معلومات دقيقة عن موضوع السؤال من مصادر جامعة {university_name} بتقولي إنه ما عندك معلومات كافية أو 'ما لقيت والله'. "
    "*** Handling Requirement Gaps (مهم!): *** "
    "إذا الطالب سأل عن شي وما حقق الشرط، شوفي قديش الفرق:"
    "   1.  **إذا الفرق بسيط (Near Miss):** زي معدل ناقص علامة أو علامتين. وضّحيله الشرط الرسمي (مثلاً 'المعدل المطلوب 65') بس بعدها ضيفي لمسة إنسانية، زي مثلاً: 'بصراحة، فرق علامة وحدة... مش عارفة إذا بمشوها أو لأ 💔. بحسها مش حجة كبيرة، بس القوانين قوانين مرات🤷‍♀️. الأحسن تتواصل مع قسم القبول والتسجيل بالجامعة نفسها {university_name} وتتأكد منهم مباشرة، بكون أفضل إشي عشان تاخد الجواب الأكيد'. (حافظي على الأمل والنصيحة بالتواصل)."
    "   2.  **إذا الفرق كبير (Far Miss):** زي معدل 60 وبدو طب (اللي بدو 85+). هنا كوني صريحة بس بطريقة ودية ومضحكة شوي. وضحي الشرط بجدية (مثلاً 'معدل الطب بدو فوق الـ 85') وبعدها علّقي عالفرق الكبير بضحكة خفيفة مع الـ skull emoji، زي مثلاً: 'ف انو معدلك 60 وبدك طب؟  💀 .او براه شو جد بتحكي . الفرق كبير بصراحة. يمكن تشوف تخصص تاني قريب أو بمجال تاني؟ في كتير شغلات حلوة كمان!'. (كوني واضحة انه صعب كتير بس بطريقة لطيفة ومضحكة 💀، واقترحي بدائل)."
    "*** End Handling Requirement Gaps ***"
    "*** Comparison Offer Instruction (مهم جداً!) ***"
    "عندما تقدمين معلومات عن **رسوم الساعات** لتخصص معين، وإذا قررتِ عرض مقارنة، يجب أن يبدأ هذا الجزء من ردكِ دائماً بـ: \n1. سطر يحتوي على الشرطات الثلاث \'---\' فقط (ليُعرض كخط فاصل أفقي).\n2. سطر فارغ بعده.\n3. ثم نص الاقتراح الذي يبدأ بـ \'🤔 على فكرة،...\'.\nمثال على بداية هذا الجزء: \'\n---\n\n🤔 على فكرة، إذا حابب، بقدر أعرضلك مقارنة لـ **اسم التخصص الذي تحدثتِ عنه للتو** بخصوص **الرسوم الدراسية (سعر الساعة)** مع باقي الجامعات اللي عنا. شو رأيك؟\'. تأكدي من استخدام النجمتين (**) للتأكيد على اسم التخصص ونوع المعلومة."

    "وعندما تقدمين معلومات عن **معدلات القبول** لتخصص معين، وإذا قررتِ عرض مقارنة، يجب أن يبدأ هذا الجزء من ردكِ دائماً بـ: \n1. سطر يحتوي على الشرطات الثلاث \'---\' فقط (ليُعرض كخط فاصل أفقي).\n2. سطر فارغ بعده.\n3. ثم نص الاقتراح الذي يبدأ بـ \'🤔 على فكرة،...\'.\nمثال على بداية هذا الجزء: \'\n---\n\n🤔 على فكرة، إذا حابب، بقدر أعرضلك مقارنة لـ **اسم التخصص الذي تحدثتِ عنه للتو** بخصوص **شروط القبول (المعدل المطلوب)** مع باقي الجامعات اللي عنا. شو رأيك؟\'. تأكدي من استخدام النجمتين (**) للتأكيد على اسم التخصص ونوع المعلومة."

    "ملاحظة: لا تعرضي مقارنة إذا لم يكن السؤال الأصلي عن رسوم أو معدلات قبول، أو إذا لم تتوفر لديك معلومات كافية عن التخصص المطلوب لتقديم إجابة أولية وافية."
    "*** End Comparison Offer Instruction ***"
    "شغلة مهمة كتير: لو لقيتي أي شي عن رسوم، سعر ساعة، أو معدل قبول (خصوصي لو بقسم '--- الرسوم ---' أو '--- شروط القبول ---')، "
    "ركزي عليها وجيبيها بالإجابة أول شي، هاي معلومات مهمة كتير للطالب. استخدمي المعلومات الإضافية بس لدعم هاي النقاط. "
    "كمان شغلة مهمة، إذا لقيتي رابط للمصدر (بيكون مكتوب 'الرابط: ...') مع المعلومة، يا ريت تذكريه كمان في جوابك عشان الطالب يقدر يشوف التفاصيل بنفسه. 👍"
    "\n\n**استخدمي تنسيق ماركداون (Markdown)** لجعل إجاباتك مرتبة وسهلة القراءة. مثلاً: استخدمي **النص العريض** للعناوين أو النقاط المهمة، والقوائم النقطية (-) أو المرقمة (1.) لتعداد المعلومات."
    "\n\n*** University Rivalry Instructions (مهم جداً!) ***"
    "عندما تتحدثين عن جامعتك {university_name} مقارنة بالجامعات الأخرى:"
    "1. **الرسوم الدراسية:** إذا كانت رسوم جامعتك أقل، علقي بشكل مرح: 'وبعدين، شو بدك أكتر من هيك؟ عنا أرخص بكتير من غيرنا! 😏' أو 'شوف كيف عنا بتوفر عليك! 💰'"
    "2. **معدلات القبول:** إذا كانت معدلات القبول في جامعتك أقل، علقي: 'وبعدين، شو بدك أكتر من هيك؟ عنا بتقبل بسهولة أكتر من غيرنا! 😉' أو 'شوف كيف عنا بتساعدك! 📚'"
    "3. **المنشآت والخدمات:** علقي على مميزات جامعتك بشكل مرح: 'وبعدين، شو بدك أكتر من هيك؟ عنا أحلى منشآت! 🏛️' أو 'شوف كيف عنا أحسن! 🌟'"
    "4. **التنافسية:** استخدمي تعابير مثل: 'وبعدين، شو بدك أكتر من هيك؟ عنا أحسن من غيرنا! 😎' أو 'شوف كيف عنا بتفوز! 🏆'"
    "5. **التميز:** علقي على تميز جامعتك: 'وبعدين، شو بدك أكتر من هيك؟ عنا أحسن من غيرنا! 🌟' أو 'شوف كيف عنا بتفوق! 🎓'"
    "*** End University Rivalry Instructions ***"
)

# Initialize session memory
session_memory: Dict[str, Dict] = {}

# Define request and response models
class AskRequest(BaseModel):
    session_id: str
    university: str
    message: str

class AskResponse(BaseModel):
    answer: str

# Initialize FastAPI
app = FastAPI(
    title="University Assistant API",
    description="API for multi-university assistant powered by AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Retrieval function that uses university as namespace
def retrieve(uni: str, query: str, k: int = 7):
    """
    Embed the user query and fetch the top‑k chunks that belong
    ONLY to `uni`, using the metadata filter.  All vectors are in
    the default namespace ("").
    """
    # Optional: Arabic -> English synonym map for common major names
    arabic_english_synonyms = {
        "كمبيوتر ساينس": "computer science",
        "علم الحاسوب": "computer science",
        "طب": "medicine",
        "هندسة": "engineering",
        "محاسبة": "accounting",
        "إدارة أعمال": "business administration",
        "تسويق": "marketing",
        "اقتصاد": "economics",
        "صيدلة": "pharmacy",
        # Add more common transcriptions/translations as needed
    }
    processed_query = query
    for ar, en in arabic_english_synonyms.items():
        # Basic replacement, consider word boundaries if needed for more complex cases
        # Use re.sub for case-insensitive replacement
        try:
             # Escape potential regex special characters in the Arabic key
             escaped_ar = re.escape(ar)
             processed_query = re.sub(escaped_ar, en, processed_query, flags=re.IGNORECASE | re.UNICODE)
        except Exception as re_err:
             logger.warning(f"Regex error during synonym replacement for '{ar}': {re_err}")
             # Fallback or skip this synonym if regex fails
             pass # Continue with the next synonym
    
    if processed_query != query:
        logger.info(f"Query processed with synonyms: Original='{query}', Processed='{processed_query}'")
    else:
        processed_query = query # Use original if no synonyms matched

    # 1 – embed once (using processed query)
    vec = openai.embeddings.create(
            model="text-embedding-3-small",
            input=processed_query # Use the processed query
          ).data[0].embedding

    # 2 – query Pinecone with case-insensitive filter
    res = index.query(
            vector=vec,
            top_k=k,
            namespace="",                 # default namespace
            filter={"university": uni.lower().strip()},   # case-insensitive filter
            include_metadata=True
          )

    matches = res.get("matches", [])
    print(f"Found {len(matches)} matches with university filter: {uni}")
    
    # Debug: print the first match ID and metadata if available
    if matches and len(matches) > 0:
        print(f"First match ID: {matches[0]['id']}")
        print(f"Metadata: {matches[0].get('metadata', {})}")
        
        # Print the raw match data to see the complete structure
        logger.info(f"Raw first match: {json.dumps(matches[0], ensure_ascii=False, default=str)}")
    else:
        logger.warning(f"No matches found for university: {uni} and query: {query}")
    
    return matches

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    logger.info(f"Received request - Session: {req.session_id}, University: {req.university}, Message: '{req.message}'")
    
    # Initialize or update session memory
    if req.session_id not in session_memory:
        session_memory[req.session_id] = {
            "uni": req.university, 
            "messages": [], 
            "summary": "", 
            "comparable_context": None,
            "navigation_history": [],
            "last_user_queries": {} # Added last_user_queries
        }
    
    mem = session_memory[req.session_id]
    
    # --- Navigation History Update ---
    # Ensure consistent casing for history and current university ID
    current_uni_id_for_history = req.university.lower() 
    nav_history = mem.get("navigation_history", [])

    # Only add to history if it's a new university in the sequence
    if not nav_history or nav_history[-1] != current_uni_id_for_history:
        nav_history.append(current_uni_id_for_history)
        mem["navigation_history"] = nav_history
        logger.info(f"Updated navigation history for session {req.session_id}: {nav_history}")
    # --- End Navigation History Update ---
    
    # If university changed (for session's primary uni tracking), reset messages, etc.
    # The navigation_history persists to allow cross-university "memory" for greetings.
    if mem["uni"] != req.university:
        logger.info(f"University changed from {mem['uni']} to {req.university}, resetting memory")
        mem["uni"] = req.university
        mem["messages"] = []
        mem["summary"] = ""
        mem["comparable_context"] = None # Reset comparable_context on uni change
    
    # Get full university name
    university_names = {
        "aaup": "الجامعة العربية الأمريكية",
        "birzeit": "جامعة بيرزيت",
        "ppu": "جامعة بوليتكنك فلسطين",
        "an-najah": "جامعة النجاح الوطنية",
        "bethlehem": "جامعة بيت لحم",
        "alquds": "جامعة القدس"
    }
    
    university_name = university_names.get(req.university, req.university)
    
    # 0) ***SAVE the current user turn *before* rewriting***
    # mem["messages"].append({"role": "user", "content": req.message}) # MOVED: User message saved after potential initial greeting

    # --- Handle Initial Greeting Request ---
    if req.message == "__INITIAL_GREETING__":
        dynamic_welcome_text = generate_dynamic_welcome_message(
            mem["navigation_history"], 
            current_uni_id_for_history, # Use the lowercased version
            university_names,
            AVAILABLE_PINECONE_NAMESPACES,
            mem.get("last_user_queries", {}) # Pass last_user_queries
        )
        # Add Sara's greeting to message history so she knows it was said
        mem["messages"].append({"role": "assistant", "content": dynamic_welcome_text})
        logger.info(f"Session {req.session_id}: Returning dynamic initial greeting for {req.university}: {dynamic_welcome_text}")
        return AskResponse(answer=dynamic_welcome_text)
    # --- End Handle Initial Greeting Request ---

    # --- Comparison Logic Handling ---
    user_wants_comparison = False
    if mem.get("comparable_context"):
        # Check if user's current message is an affirmative response to a comparison offer
        affirmative_responses = [
            # English affirmative responses
            "yes", "yeah", "yep", "sure", "ok", "okay", "alright", "absolutely", "definitely",
            "go ahead", "proceed", "show me", "do it", "sounds good", "why not", "of course",
            "certainly", "please", "please do", "i'd like that", "i want that", "good idea",
            "great", "excellent", "perfect", "that works", "let's see it", "i'm interested",
            "go for it", "make it happen", "let's do it", "sounds interesting", "tell me more",
            
            # Arabic standard affirmatives
            "نعم", "أجل", "بالتأكيد", "بالطبع", "موافق", "موافقة", "حسناً", "حسنا", 
            
            # Palestinian/Levantine dialect affirmatives
            "اه", "أه", "آه", "ايوا", "إيوا", "ايوه", "أيوه", "أيوة", "اي", "أي",
            "اكيد", "أكيد", "تمام", "ماشي", "طيب", "زبط", "زابط", "منيح", "مليح","اه بحب", "أوك", "أوكي", "اوكي", "اوك",
            # Action requests in Arabic
            "اعملي", "سوي", "اعرضي", "ورينا", "فرجيني", "اطلعي", "ورجيني", "قارني",
            "قارنيلي", "اعملي مقارنة", "سويلي مقارنة", "اعرضيلي", "جيبيلي", "جيبي",
            "شوفيلي", "شوفي", "احسبيلي", "احسبي", "قوليلي", "قولي", "بينيلي", "بيني",
            # Additional Palestinian/Levantine affirmative action phrases
            "اه فرجيني", "اه اعطيني", "فرجيني", "هاتي لنشوف", "فرجينا", "اعطيني", 
            "هاتي", "ورجيني", "ورينا", "هاتيلي", "هاتيلنا", "فرجيلي", "فرجيلنا",
            # Polite requests in Arabic
            "من فضلك", "لو سمحتي", "لو سمحت", "بليز", "إذا ممكن", "اذا ممكن", "إذا بتقدري",
            "اذا بتقدري", "ممكن", "يا ريت", "ياريت", "بعد إذنك", "بعد اذنك",
            
            # Desire expressions in Arabic
            "بحب", "حابب", "حاب", "اريد", "أريد", "بدي", "نفسي", "ودي", "رح اكون ممنون",
            "رح اكون ممنونة", "بتمنى", "اتمنى", "أتمنى", "محتاج", "محتاجة",
            
            # Positive feedback in Arabic
            "جيد", "حلو", "ممتاز", "رائع", "كويس", "منيح", "فكره حلوه", "فكرة حلوة",
            "عظيم", "مية مية", "مئة مئة", "١٠٠٪", "100%", "تمام التمام", "عال العال",
            
            # Compound phrases
            "اه بدي", "نعم بليز", "اكيد لو سمحتي", "طبعا اعرضي", "اه ورجيني", "نعم اكيد",
            "اه منيح", "تمام جيد", "ماشي حلو", "اي اكيد", "اه طبعا", "بالتأكيد اعرضي",
            "يلا ورجيني", "يالله اعملي", "يالله سوي", "هيا اعرضي", "هيا ورينا"
        ]
        # Normalize user message: remove punctuation, convert to lowercase
        normalized_message = req.message.lower().replace('.', '').replace('،', '').replace('؟', '').replace('!', '').strip()
        
        # More robust check:
        # 1. Check for exact matches from the list.
        # 2. Check if the normalized message *is* one of the short affirmative terms (e.g., "اه", "نعم").
        # 3. Check if the normalized message *starts with* or *contains* slightly longer affirmative phrases.
        
        user_confirmed = False
        if normalized_message in affirmative_responses: # Handles single-word exact matches like "اه"
            user_confirmed = True
        else:
            for term in affirmative_responses:
                # Check if the message IS the term, or contains it as a whole word,
                # or if the term is a multi-word phrase contained in the message.
                # This avoids partial matches like "information" matching "اه" in "information".
                if f" {term} " in f" {normalized_message} " or \
                   normalized_message.startswith(term + " ") or \
                   normalized_message.endswith(" " + term) or \
                   normalized_message == term: # Exact match after normalization
                    user_confirmed = True
                    break
        
        if user_confirmed:
            user_wants_comparison = True
            logger.info(f"User confirmed desire for comparison with message: '{req.message}' (Normalized: '{normalized_message}')")
        # else: # User might be asking a new question, so clear the offer implicitly
            # mem["comparable_context"] = None # Clear if not a direct 'yes'
            # logger.info("User did not directly affirm comparison, comparable_context cleared.")

    # If user wants comparison, skip retrieval and go to generation (will be handled later)
    if user_wants_comparison and mem["comparable_context"]:
        # Placeholder: Call a new function to generate and return comparison table
        # This will be implemented in the next step.
        # For now, we'll just log and prepare a placeholder answer.
        major_name_to_compare = mem["comparable_context"]["major_name"]
        info_type_to_compare = mem["comparable_context"]["info_type"]
        logger.info(f"Proceeding to generate comparison table for {major_name_to_compare} - {info_type_to_compare}.")
        
        # Simulate calling the comparison generation function
        # In a real scenario, this function would query all universities and format a table
        comparison_table_md = generate_comparison_table_data(
            major_name_to_compare,
            info_type_to_compare,
            list(UNIVERSITY_MAP.keys()), # Pass all university IDs
            req.university # Current university to highlight or skip
        )

        final_answer = comparison_table_md # The table itself is the answer
        
        # Add the comparison table to messages for context, then clear the flag
        mem["messages"].append({"role": "user", "content": req.message}) # User's 'yes'
        mem["messages"].append({"role": "assistant", "content": final_answer})
        mem["comparable_context"] = None # Clear after providing comparison
        logger.info("Comparison table provided and comparable_context cleared.")
        return {"answer": final_answer}
    # --- End Comparison Logic Handling ---

    # 1) Rewrite query only if we now have previous context
    try:
        # NEW: Rewrite based on history *before* adding the current message
        if len(mem["messages"]) > 0:          # Check if there's *any* history
            # Filter history for actual user/assistant turns, excluding tool messages
            # AND excluding our internal query display messages
            history_for_rewrite = [
                m for m in mem["messages"] 
                if m["role"] in ("user", "assistant") and 
                not m.get("content", "").startswith("🔎 استعلام داخلي:") # Exclude internal query logs
            ]
            standalone_query = rewrite_query(history_for_rewrite,
                                             req.message,         # Pass current message separately
                                             university_name)
            logger.info(f"Original: '{req.message}' | Rewritten for retrieval: '{standalone_query}'")
        else:
            standalone_query = req.message # Use original query if no history
            logger.info(f"No history, using original query for retrieval.")
        # --- End rewrite ---

        # ***SAVE the current user turn *after* rewriting***
        mem["messages"].append({"role": "user", "content": req.message})

        matches = retrieve(mem['uni'], standalone_query)
        mem["comparable_context"] = None # Reset context before potentially setting it again
        # --- End rewrite ---

        # --- Variable to store identified major and info type for comparison offer ---
        identified_major_for_comparison: Optional[str] = None
        identified_info_type_for_comparison: Optional[str] = None
        # --- End Variable --- 

        # Handle empty matches case
        if not matches:
            context = f"لا توجد معلومات متاحة حاليًا عن {university_name}. أنا سارة، بدي أذكرك إنه هذه منصة تجريبية وما زلنا نضيف المعلومات عن الجامعات."
            logger.warning(f"No matches found for {req.university}, using default context")
        else:
            # Extract context from matches - handle different response formats
            try:
                context_parts = []
                price_info = ""
                admission_info = ""
                found_price_in_context = False
                
                logger.info(f"Processing {len(matches)} matches to extract context")
                
                # --- Refined Context Extraction Logic ---
                for i, m in enumerate(matches):
                    # --- Attempt to access data assuming dict-like or object structure ---
                    try:
                        # Use .get for dicts, getattr for objects, provide defaults
                        match_id = m.get('id', None) if isinstance(m, dict) else getattr(m, 'id', None)
                        score = m.get('score', 0.0) if isinstance(m, dict) else getattr(m, 'score', 0.0)
                        metadata = m.get('metadata', {}) if isinstance(m, dict) else getattr(m, 'metadata', {})

                        # If getattr returned the default {} and id is None, the object structure is unexpected
                        if match_id is None and metadata == {}:
                             logger.warning(f"Match {i}: Could not extract id or metadata. Match type: {type(m)}, skipping.")
                             continue
                        if match_id is None:
                             match_id = f"unknown_match_{i}" # Assign placeholder if ID is missing

                        # Ensure metadata is a dictionary after retrieval
                        if not isinstance(metadata, dict):
                             # If metadata is a string, try parsing it as JSON (common issue)
                             if isinstance(metadata, str):
                                 try:
                                     metadata = json.loads(metadata)
                                     if not isinstance(metadata, dict):
                                          logger.warning(f"Match {i} ({match_id}): Parsed metadata is not a dictionary ({type(metadata)}), skipping.")
                                          continue
                                     logger.info(f"Match {i} ({match_id}): Successfully parsed metadata string into dict.")
                                 except json.JSONDecodeError:
                                     logger.warning(f"Match {i} ({match_id}): Metadata is a non-JSON string, skipping. Content: {metadata[:100]}...")
                                     continue
                             else:
                                logger.warning(f"Match {i} ({match_id}): Metadata is not a dictionary or string ({type(metadata)}), skipping.")
                                continue

                        # Ensure metadata dict is not empty before proceeding
                        if not metadata:
                             logger.warning(f"Match {i} ({match_id}): Metadata dictionary is empty, skipping.")
                             continue

                        logger.info(f"--- Processing Match {i} (ID: {match_id}, Score: {score:.4f}) ---")

                    except (AttributeError, TypeError, KeyError) as e:
                        logger.warning(f"Match {i}: Error accessing standard fields (id, score, metadata) - Skipping. Error: {e}. Match data: {str(m)[:100]}...")
                        continue
                        
                    # --- The rest of the extraction logic using the retrieved 'metadata' dict ---

                    # --- 1. Prioritize Extracting Fee Information ---
                    fee_part_extracted = None # To store the found fee value
                    # Check multiple potential fields for fee info
                    text_to_search_for_fee = ""
                    if 'text' in metadata and isinstance(metadata['text'], str):
                        text_to_search_for_fee = metadata['text']
                    elif 'original_text' in metadata and isinstance(metadata['original_text'], str):
                        text_to_search_for_fee = metadata['original_text']
                    metadata_str_lower = json.dumps(metadata, ensure_ascii=False, default=str).lower()

                    fee_found_in_match = False
                    # Pattern 1: "Credit-hour fee: 350"
                    fee_match_eng = re.search(r'credit-hour fee:?\s*(\d+)', text_to_search_for_fee.lower())
                    if not fee_match_eng: fee_match_eng = re.search(r'credit-hour fee:?\s*(\d+)', metadata_str_lower)
                    if fee_match_eng:
                        fee_part_extracted = fee_match_eng.group(1)
                        fee_found_in_match = True

                    # Pattern 2: "رسوم الساعة: 70"
                    if not fee_found_in_match:
                        fee_match_ar = re.search(r'رسوم الساعة[^٠-٩]*([٠-٩]+|[0-9]+)', text_to_search_for_fee)
                        if not fee_match_ar: fee_match_ar = re.search(r'رسوم الساعة[^٠-٩]*([٠-٩]+|[0-9]+)', metadata_str_lower)
                        if fee_match_ar:
                            fee_part_extracted = fee_match_ar.group(1)
                            fee_found_in_match = True

                    # Store the first fee found globally for the request
                    if fee_found_in_match and not price_info: # Only store the first fee encountered
                        price_info = f"💰 سعر الساعة المعتمدة هو {fee_part_extracted} شيكل أو دينار حسب عملة الجامعة [{metadata.get('title', 'المصدر')}]."
                        logger.info(f"Stored Price Info from match {i}: {price_info}")
                        found_price_in_context = True

                    # --- 2. Extract Text Content for General Context & Admission Info ---
                    extracted_text = ""
                    # admission_part_extracted = None # Removed this global flag for the loop iteration

                    potential_text_fields = ['text', 'original_text', 'content', 'description']

                    for field_name in potential_text_fields:
                        if field_name in metadata:
                            text_content = metadata.get(field_name)
                            current_text = ""
                            if isinstance(text_content, list):
                                current_text = ' '.join(filter(None, [str(item) for item in text_content]))
                            elif isinstance(text_content, str):
                                current_text = text_content
                            elif text_content is not None:
                                current_text = str(text_content)

                            # Example: current_text = metadata.get(field_name, '')
                            # Ensure current_text is a string
                            if not isinstance(current_text, str):
                                current_text = str(current_text) if current_text is not None else ""

                            if current_text.strip():
                                extracted_text = current_text # Keep the full text for context
                                # logger.info(f"Found text in field '{field_name}' for match {i}") # Keep logging minimal now

                                # --- 2a. Look for Admission Average within this text ---
                                if not admission_info: # Only store the first admission average found *globally* across all matches
                                    admission_part_extracted_from_this_match = None # Reset check for each match's text
                                    # Pattern 1: "65 Admission: ..."
                                    adm_match_eng1 = re.search(r'(\d{2,3})\s+Admission:', extracted_text)
                                    # Pattern 2: "... Admission: 65 ..."
                                    adm_match_eng2 = re.search(r'Admission:\s*(\d{2,3})', extracted_text, re.IGNORECASE)
                                    # Pattern 3: "معدل القبول: 65"
                                    adm_match_ar = re.search(r'معدل القبول[^\d]*(\d{2,3})', extracted_text)
                                    # Pattern 4: Handles "Admission: [text]\n70" format
                                    adm_match_newline = re.search(r'Admission:[^\n]*\n\s*(\d{2,3})', extracted_text)

                                    # Check patterns in order of preference/specificity
                                    if adm_match_newline:
                                        admission_part_extracted_from_this_match = adm_match_newline.group(1)
                                        # logger.debug(f"Match {i}: Found admission rate via newline pattern: {admission_part_extracted_from_this_match}")
                                    elif adm_match_eng1:
                                        admission_part_extracted_from_this_match = adm_match_eng1.group(1)
                                        # logger.debug(f"Match {i}: Found admission rate via eng1 pattern: {admission_part_extracted_from_this_match}")
                                    elif adm_match_eng2:
                                        admission_part_extracted_from_this_match = adm_match_eng2.group(1)
                                        # logger.debug(f"Match {i}: Found admission rate via eng2 pattern: {admission_part_extracted_from_this_match}")
                                    elif adm_match_ar:
                                        admission_part_extracted_from_this_match = adm_match_ar.group(1)
                                        # logger.debug(f"Match {i}: Found admission rate via arabic pattern: {admission_part_extracted_from_this_match}")

                                    if admission_part_extracted_from_this_match:
                                        # If found in this match's text AND global admission_info isn't set yet, store it.
                                        admission_info = f"ℹ️ معدل القبول المطلوب هو حوالي {admission_part_extracted_from_this_match}% [{metadata.get('title', 'المصدر')}]."
                                        logger.info(f"Stored Admission Info from match {i} (Value: {admission_part_extracted_from_this_match}): {admission_info}")

                                # Once text is found (regardless of admission found), break inner loop over fields
                                break # This breaks the loop over potential_text_fields

                    # Clean the extracted text (only if found)
                    if extracted_text:
                        cleaned_log_text = "(Cleaning failed)" # Initialize for the case where try fails
                        try:
                            # Minimal cleaning: normalize whitespace
                            extracted_text = ' '.join(extracted_text.split())

                            # Log the potentially readable text snippet
                            log_snippet = extracted_text[:150].replace('\\n', ' ') # Show first 150 chars, replace newlines for logging
                            logger.info(f"Cleaned text snippet from match {i}: {log_snippet}...")
                            cleaned_log_text = extracted_text # Use the cleaned text if successful

                        except Exception as clean_error:
                            logger.warning(f"Could not clean text from match {i}: {clean_error}. Raw text snippet: {extracted_text[:100]}...")
                            extracted_text = "" # Discard if cleaning fails badly
                            cleaned_log_text = "(Cleaning failed)"

                    # --- 3. Construct General Context Part ---
                    # Use the potentially cleaned extracted_text (or original if cleaning failed but still usable)
                    # Only add if we have meaningful text
                    if extracted_text and len(extracted_text.strip()) > 10:
                        # Build the title/source string
                        title = metadata.get('title', '').strip()
                        section = metadata.get('section', '').strip()
                        source_name = title
                        if section and section.lower() != title.lower(): # Avoid "Title (Title)"
                            source_name = f"{title} ({section})"
                        if not source_name:
                            source_name = f"مصدر {i+1}" # Fallback source name

                        # Add URL if available
                        url = metadata.get('url', '')
                        url_ref = f" (الرابط: {url})" if url else ""

                        # Limit text length to avoid excessive context, increased limit
                        display_text = extracted_text[:800]
                        if len(extracted_text) > 800:
                             display_text += "..."

                        context_part = f"[{source_name}] {display_text}{url_ref}"
                        context_parts.append(context_part)
                        # logger.info(f"Added context part {len(context_parts)} from match {i}: {context_part[:100]}...") # Keep logging minimal
                    else:
                         # logger.info(f"Skipping context part for match {i} due to lack of usable text.") # Keep logging minimal
                         pass # No action needed if text is not usable

                # --- End of Loop --- (This is the end of the main `for i, m in enumerate(matches):` loop)

                # --- 4. Combine Final Context with Structure ---
                final_context_parts = []

                # Add prioritized price info if found
                if price_info:
                    final_context_parts.append("--- الرسوم ---")
                    final_context_parts.append(price_info)
                    logger.info("Adding Price section to context.")

                # Add prioritized admission info if found
                if admission_info:
                     final_context_parts.append("--- شروط القبول ---")
                     final_context_parts.append(admission_info)
                     logger.info("Adding Admission section to context.")

                # Add the general context parts, potentially filtered
                if context_parts:
                    final_context_parts.append("--- معلومات إضافية --- ")
                    # Join general context with newlines
                    final_context_parts.append("\n".join(context_parts))
                    logger.info(f"Adding {len(context_parts)} general context parts.")

                # ---- Identify potential major name and info type from matches for comparison ----
                if matches and len(matches) > 0:
                    first_match_metadata = matches[0].get('metadata', {}) if isinstance(matches[0], dict) else getattr(matches[0], 'metadata', {})
                    if isinstance(first_match_metadata, str): # try to parse if string (shouldn't happen with current retrieve logic)
                        try: first_match_metadata = json.loads(first_match_metadata)
                        except: first_match_metadata = {}
                    
                    if isinstance(first_match_metadata, dict):
                        potential_major_title = first_match_metadata.get('title')
                        if potential_major_title and isinstance(potential_major_title, str):
                            # Heuristic to check if the query was about this major's fee or admission
                            query_lower = standalone_query.lower()
                            is_fee_query = any(term in query_lower for term in ["fee", "price", "cost", "سعر", "تكلفة", "رسوم"])
                            is_admission_query = any(term in query_lower for term in ["admission", "average", "avg", "rate", "معدل", "قبول"])

                            if is_fee_query:
                                identified_major_for_comparison = potential_major_title
                                identified_info_type_for_comparison = "الرسوم الدراسية (سعر الساعة)"
                                logger.info(f"Identified fee query for major: {potential_major_title}")
                            elif is_admission_query:
                                identified_major_for_comparison = potential_major_title
                                identified_info_type_for_comparison = "شروط القبول (المعدل المطلوب)"
                                logger.info(f"Identified admission query for major: {potential_major_title}")
                # ---- End Identification ----

                # Check if we actually have any context to show
                if len(final_context_parts) > 0:
                    context = "\n\n".join(final_context_parts) # Use double newline between sections
                    logger.info(f"Successfully built structured context.")
                elif matches: # Matches were found, but extraction yielded nothing useful
                    logger.warning("Matches found but no usable context could be extracted. Creating fallback.")
                    match_ids = [getattr(m, 'id', f'match_{idx}') for idx, m in enumerate(matches[:3])]
                    context = (f"لقد وجدت بعض المعلومات المتعلقة بسؤالك في مصادر {university_name} "
                               f"(مثل: {', '.join(match_ids)}), ولكن لم أتمكن من استخلاص التفاصيل بوضوح. "
                               "قد تحتاج إلى مراجعة المصادر مباشرة أو إعادة صياغة سؤالك.")
                else: # No matches were found initially
                    logger.warning(f"No matches found for '{req.message}' in {university_name}. Using 'no info' context.")
                    context = f"بصراحة يا صاحبي، ما لقيت معلومات كافية عن سؤالك بخصوص '{req.message}' في بيانات {university_name} المتوفرة عندي حالياً 🤷‍♀️."


                logger.info(f"Final context length: {len(context)} characters")
                # logger.debug(f"Final Context:\n{context}") # Uncomment for deep debug

            except Exception as ctx_error:
                logger.error(f"Critical error during context extraction: {str(ctx_error)}", exc_info=True)
                context = "حدث خطأ فني أثناء محاولة استخلاص المعلومات، بعتذر منك 🙏. ممكن تجرب تسأل مرة ثانية؟"
        
        # Build prompt with Sara persona, memory summary, and context
        formatted_sara_prompt = SARA_PROMPT.format(university_name=university_name)
        
        # Detect if it's a price question
        is_price_question = any(term in req.message.lower() for term in 
                               ["سعر", "تكلفة", "رسوم", "شيكل", "دينار", "قديش", "كم", "price", "fee", "cost", "tuition"]) # Added more terms
        
        if is_price_question:
            logger.info("Detected price-related question.")
            # Special instruction is now primarily handled by prepending price_info to context
            # Optional: Add a subtle reminder if needed, but avoid redundancy
            # price_instruction = "\n\n(تذكير: ركزي على معلومات السعر إذا كانت متوفرة)"
            price_instruction = "" # Keep it clean, rely on context structure
        else:
            price_instruction = ""
        
        # Get response from OpenAI
        try:
            # Create a better structured prompt with context
            formatted_sara_prompt = SARA_PROMPT.format(university_name=university_name)
            
            prompt_construction_parts = [formatted_sara_prompt]

            # History for prompt context should be messages *before* the current user's latest message in mem["messages"].
            # mem["messages"] includes the current user's message as the last element at this stage.
            history_for_prompt_context = mem["messages"][:-1] 

            if mem.get('summary'): # Use .get to avoid KeyError if summary not initialized
                prompt_construction_parts.append(f"\n\nملخص المحادثة السابقة:\n{mem['summary']}")
            elif history_for_prompt_context: # If no long-term summary, but there's immediate history
                # Create a concise representation of recent history (e.g., last 2 turns = up to 4 messages)
                recent_history_lines = []
                # Show up to last 2 user messages and 2 assistant responses
                for m in history_for_prompt_context[-4:]: 
                    role_display = "أنت (المستخدم)" if m['role'] == 'user' else "أنا (سارة)"
                    # Limit length of each message content in the snippet
                    content_snippet = m['content'][:150] + "..." if len(m['content']) > 150 else m['content']
                    recent_history_lines.append(f"{role_display}: {content_snippet}")
                
                if recent_history_lines:
                    prompt_construction_parts.append(f"\n\nمقتطف من المحادثة الجارية:\n" + "\n".join(recent_history_lines))
            
            # Add context header to clarify the source of information
            prompt_construction_parts.append(f"\n\n--- معلومات من {university_name} ---\n{context}")
            
            # Add the actual question
            prompt_construction_parts.append(f"\n\n--- السؤال ---\n{req.message}")

            prompt = "".join(prompt_construction_parts)
            
            # Log the prompt structure
            logger.info(f"Prompt structure: Sara persona + context ({len(context)} chars) + question")
            logger.info(f"Final context length used: {len(context)} characters")
            
            # Store message in memory
            # mem["messages"].append({"role": "user", "content": req.message}) # <-- REDUNDANT, REMOVED
            
            # --- Refined Message Construction for LLM ---
            messages = []
            # Start with the system prompt (persona + context + question structure)
            messages.append({"role": "system", "content": prompt})
            
            # Add the synthetic assistant message showing the internal query - Role reverted back to assistant
            messages.append({"role": "assistant", "content": f"🔎 استعلام داخلي: {standalone_query}"}) # Reverted role to assistant

            # Add the user's direct question
            messages.append({"role": "user", "content": req.message})
            
            # If it's a price question AND we extracted specific price info,
            # we can optionally add an assistant pre-fill to guide the model,
            # but often just having the price clearly in the context is enough.
            # Example of pre-filling (use with caution, might make responses too rigid):
            # if is_price_question and price_info:
            #    logger.info("Adding price guidance message based on extracted info.")
            #    messages.append({"role": "assistant", "content": f"بالنسبة للسعر، {price_info}"}) # Start the answer

            logger.info(f"Sending {len(messages)} messages to OpenAI API. System prompt includes context length: {len(context)}")
            # logger.debug(f"Messages sent to OpenAI: {messages}") # Optional: log the exact messages
            
            response = openai.chat.completions.create(
                model="gpt-4-turbo", # Or your preferred model
                messages=messages,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Generated response for session {req.session_id}: {answer[:200]}")

            # Try to parse comparison offer from Sara's response
            # Example offer format: "---🤔 على فكرة، إذا حابب، بقدر أعرضلك مقارنة لـ **MAJOR_NAME** بخصوص **INFO_TYPE** مع باقي الجامعات اللي عنا. شو رأيك؟"
            # Regex captures MAJOR_NAME and the specific INFO_TYPE string Sara was instructed to use.
            offer_match = re.search(r"مقارنة لـ \*\*(.+?)\*\* بخصوص \*\*(الرسوم الدراسية \(سعر الساعة\)|شروط القبول \(المعدل المطلوب\))\*\*", answer, re.DOTALL)

            if offer_match:
                offered_major = offer_match.group(1).strip()
                standardized_info_type = offer_match.group(2).strip() # This will be one of the two exact strings
                
                mem["comparable_context"] = {
                    "major_name": offered_major,
                    "info_type": standardized_info_type
                }
                logger.info(f"Comparison offer parsed from LLM response: Major='{offered_major}', InfoType='{standardized_info_type}'. Context stored.")
            # If no offer is parsed, mem["comparable_context"] remains as it was (cleared at the start of the request)

            # Update memory with the answer
            mem["messages"].append({"role": "assistant", "content": answer})
            
            # If memory gets too long, summarize it
            if len(mem["messages"]) > 10:
                logger.info(f"Summarizing conversation for session {req.session_id}")
                summary_response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Summarize the following conversation in Arabic:"},
                        {"role": "user", "content": str(mem["messages"])}
                    ]
                )
                mem["summary"] = summary_response.choices[0].message.content
                mem["messages"] = mem["messages"][-4:]  # Keep only recent messages
            
            logger.info(f"Final Answer returned: {answer[:500]}...") # Log the first 500 chars of the final answer
            return {"answer": answer}
        
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        # Check if Pinecone index is accessible
        stats = index.describe_index_stats()
        
        # Extract namespace information directly
        namespace_info = []
        if hasattr(stats, 'namespaces'):
            namespace_info = list(stats.namespaces.keys())
        
        return {
            "status": "healthy", 
            "pinecone": "connected", 
            "openai": "ready",
            "namespaces": namespace_info
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/load-sample-data")
async def load_sample_data():
    """Endpoint to load sample data into Pinecone for testing."""
    try:
        # Import the upload_sample_data function
        import upload_sample_data
        
        result = upload_sample_data.upload_sample_data()
        if result:
            return {"status": "success", "message": "Sample data loaded successfully"}
        else:
            return {"status": "error", "message": "Failed to load sample data"}
    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")
        return {"status": "error", "message": f"Error: {str(e)}"}

# --- Add Match Majors Endpoint ---
@app.post("/match-majors", response_model=List[Major])
async def match_majors(req: MatchMajorsRequest):
    """Filters majors based on university, average, branch, and field."""
    logger.info(f"Received /match-majors request: Uni='{req.university}', Avg={req.min_avg}, Branch='{req.branch}', Field='{req.field}'")

    if not majors_data:
        logger.error("Majors data is not loaded. Cannot process /match-majors.")
        raise HTTPException(status_code=500, detail="Server error: Major data not available.")

    matched_majors = []
    parsed_count = 0
    filter_count = 0

    for major_dict in majors_data:
        # 1. Filter by university first (case-insensitive)
        if major_dict.get('university', '').lower() != req.university.lower():
            continue

        try:
            # 2. Parse details (fee, avg, branches, field)
            parsed_major = parse_major_details(major_dict)
            parsed_count += 1
        except Exception as e:
            logger.warning(f"Failed to parse major {major_dict.get('id', 'unknown')}: {e}. Skipping.")
            continue

        # 3. Apply filters
        passes_filter = True

        # Filter by minimum average
        if req.min_avg is not None:
            if parsed_major.parsed_min_avg is None: # Cannot compare if major has no avg info
                passes_filter = False
            elif req.min_avg < parsed_major.parsed_min_avg: # Corrected comparison
                # logger.debug(f"Filtering out {parsed_major.id}: User avg {req.min_avg} < Major avg {parsed_major.parsed_min_avg}")
                passes_filter = False

        # Filter by branch
        if passes_filter and req.branch is not None and req.branch:
            # Check if the required branch is acceptable for the major
            # Major must accept "جميع أفرع التوجيهي" OR the specific branch
            if not ("جميع أفرع التوجيهي" in parsed_major.parsed_branches or req.branch in parsed_major.parsed_branches):
                # logger.debug(f"Filtering out {parsed_major.id}: User branch '{req.branch}' not in {parsed_major.parsed_branches}")
                passes_filter = False

        # Filter by field (using parsed_field)
        if passes_filter and req.field is not None and req.field:
            # Perform case-insensitive comparison
            if parsed_major.parsed_field is None or parsed_major.parsed_field.lower() != req.field.lower():
                 passes_filter = False
                 # logger.debug(f"Filtering out {parsed_major.id}: Major field '{parsed_major.parsed_field}' != Req field '{req.field}'")

        if passes_filter:
            matched_majors.append(parsed_major)
            filter_count += 1

    logger.info(f"Parsed {parsed_count} majors for {req.university}. Found {filter_count} matches after filtering.")
    return matched_majors
# --- End Match Majors Endpoint ---

# Pydantic model for the request body
class GenerateDescriptionRequest(BaseModel):
    title: str = Field(..., example="Computer Science")
    university_name: str | None = Field(None, example="Example University")

# Pydantic model for the response body
class GenerateDescriptionResponse(BaseModel):
    description: str

@app.post("/api/generate-description", response_model=GenerateDescriptionResponse, tags=["AI Generation"])
async def generate_major_description(request: GenerateDescriptionRequest):
    """
    Generates a brief description for a given major title using OpenAI.
    """
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured on server.")

    # Request a shorter description (20-30 words) in Arabic with Gen Z style
    prompt = f"اكتب وصف قصير وجذاب (حوالي 20-30 كلمة) بأسلوب شبابي عصري عن تخصص \"{request.title}\""
    if request.university_name:
        prompt += f" في {request.university_name}"
    prompt += ". ركز على المواضيع الأساسية والفرص الوظيفية المستقبلية. استخدم لغة واضحة ومفهومة وبأسلوب يناسب جيل Z - خليها كول وعفوية ومباشرة 🔥"

    try:
        logger.info(f"Generating description for: {request.title} (Uni: {request.university_name})")
        # Using the newer OpenAI client syntax for chat completions
        client = openai.OpenAI(api_key=openai.api_key) # Create client instance
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo", # Or use a more advanced model if preferred
            messages=[
                {"role": "system", "content": "انت مساعد بتساعد الطلاب يفهموا تخصصات الجامعة. استخدم لغة شبابية عصرية ومباشرة 🔥"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100, # Limit response length
            temperature=0.7, # Balance creativity and focus
            n=1,
            stop=None,
        )
        
        # Ensure choices are available and contain message content
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            description = completion.choices[0].message.content.strip()
            logger.info(f"Generated description: {description}")
            return GenerateDescriptionResponse(description=description)
        else:
             logger.error("OpenAI response was empty or malformed.")
             raise HTTPException(status_code=500, detail="Failed to generate description: Empty response from AI.")

    except openai.AuthenticationError:
        logger.error("OpenAI Authentication Error: Check API key.")
        raise HTTPException(status_code=500, detail="AI service authentication failed.")
    except openai.RateLimitError:
        logger.warning("OpenAI Rate Limit Exceeded.")
        raise HTTPException(status_code=429, detail="Rate limit exceeded for AI service. Please try again later.")
    except Exception as e:
        logger.error(f"Error generating description: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the description: {e}")

# --- Generate Comparison Table Function ---
def generate_comparison_table_data(major_name: str, info_type: str, all_university_ids: List[str], current_university_id: str) -> str:
    """Generates a Markdown table comparing a major's info across universities."""
    logger.info(f"Generating comparison for Major: '{major_name}', Info: '{info_type}' across {len(all_university_ids)} unis.")

    if not majors_data:
        logger.error("Majors data not loaded, cannot generate comparison.")
        return "اعتذر، لا يمكنني إنشاء المقارنة حاليًا بسبب عدم توفر بيانات التخصصات."

    headers = ["الجامعة", info_type, "ملاحظات"]
    rows = []

    # Get full university names for display
    university_display_names = {
        "aaup": "العربية الأمريكية",
        "birzeit": "بيرزيت",
        "ppu": "بوليتكنك فلسطين",
        "an-najah": "جامعة النجاح الوطنية",
        "bethlehem": "بيت لحم",
        "alquds": "جامعة القدس"
    }

    for uni_id in all_university_ids:
        uni_display_name = university_display_names.get(uni_id, uni_id)
        found_major_at_uni = False
        info_value = "غير متوفر"
        notes = ""

        for major_dict in majors_data:
            if major_dict.get('university', '').lower() == uni_id.lower():
                # Simple title match (case-insensitive, substring) - can be improved
                # Normalize titles slightly by removing common prefixes/suffixes if needed
                major_title_in_data = major_dict.get('title', '').lower().strip()
                query_major_name_lower = major_name.lower().strip()
                
                # Example: If major_name is "Computer Science" and title in data is "BSc Computer Science"
                if query_major_name_lower in major_title_in_data or major_title_in_data in query_major_name_lower:
                    try:
                        parsed_major = parse_major_details(major_dict.copy()) # Use a copy to avoid modifying original
                        found_major_at_uni = True

                        if "رسوم" in info_type: # Check for fee
                            if parsed_major.parsed_fee is not None:
                                currency_str = f" {parsed_major.parsed_currency}" if parsed_major.parsed_currency else ""
                                info_value = f"{parsed_major.parsed_fee}{currency_str}"
                            else:
                                info_value = "لم يتم تحديد الرسوم"
                        elif "قبول" in info_type: # Check for admission average
                            if parsed_major.parsed_min_avg is not None:
                                info_value = f"{parsed_major.parsed_min_avg}%"
                                if parsed_major.parsed_branches:
                                    notes = f"الأفرع: {', '.join(parsed_major.parsed_branches)}"
                                else:
                                    notes = "لم تحدد الأفرع"
                            else:
                                info_value = "لم يحدد المعدل"
                        
                        # Highlight current university
                        if uni_id == current_university_id:
                             uni_display_name = f"📍 {uni_display_name} (الحالية)"
                        break # Found major for this uni, move to next uni
                    except Exception as e:
                        logger.warning(f"Error parsing major {major_dict.get('id')} for {uni_id} during comparison: {e}")
                        info_value = "خطأ في المعالجة"
                        break
        
        if not found_major_at_uni:
            notes = f"لم يتم العثور على تخصص '{major_name}' بهذه الجامعة أو تفاصيله غير متاحة."
            if uni_id == current_university_id:
                 uni_display_name = f"📍 {uni_display_name} (الحالية)"

        rows.append([uni_display_name, info_value, notes])

    # Construct Markdown table
    table = f"**مقارنة {info_type} لتخصص \"{major_name}\" عبر الجامعات:**\n\n"
    table += "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---" for _ in headers]) + " |\n"
    for row_data in rows:
        table += "| " + " | ".join(str(item) for item in row_data) + " |\n"
    
    table += "\n*ملاحظة: هذه البيانات هي لأغراض المقارنة وقد تحتاج إلى تأكيد من الجامعة مباشرة.*"
    logger.info(f"Generated comparison table:\n{table}")
    return table
# --- End Generate Comparison Table Function ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 
