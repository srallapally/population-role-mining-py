# pipeline-service/config.py
import os

# --- ES connection ---
ES_HOST = os.environ.get("ES_HOST", "http://localhost:9200")
ES_API_KEY = os.environ.get("ES_API_KEY", None)
ES_SCROLL_SIZE = int(os.environ.get("ES_SCROLL_SIZE", 1000))
ES_SCROLL_TIMEOUT = os.environ.get("ES_SCROLL_TIMEOUT", "2m")

# --- Index names ---
INDEX_IDENTITIES = os.environ.get("INDEX_IDENTITIES", "mock_identities")
INDEX_ENTITLEMENTS = os.environ.get("INDEX_ENTITLEMENTS", "mock_entitlements")
INDEX_SESSIONS = os.environ.get("INDEX_SESSIONS", "mock_sessions")
INDEX_ROLES = os.environ.get("INDEX_ROLES", "mock_roles")

# --- Field mapping: identity index ---
FIELD_USER_ID = os.environ.get("FIELD_USER_ID", "userId")
FIELD_ACCOUNT_STATUS = os.environ.get("FIELD_ACCOUNT_STATUS", "accountStatus")
FIELD_ACCOUNT_TYPE = os.environ.get("FIELD_ACCOUNT_TYPE", "accountType")
FIELD_ASSIGNMENTS = os.environ.get("FIELD_ASSIGNMENTS", "assignments")
FIELD_ASSIGNMENT_ENT_ID = os.environ.get("FIELD_ASSIGNMENT_ENT_ID", "entitlementId")

# --- System filter values ---
SYSTEM_FILTER_STATUS_VALUE = os.environ.get("SYSTEM_FILTER_STATUS_VALUE", "active")
SYSTEM_FILTER_TYPE_VALUE = os.environ.get("SYSTEM_FILTER_TYPE_VALUE", "human")

# --- Field mapping: entitlement index ---
FIELD_ENT_ID = os.environ.get("FIELD_ENT_ID", "entitlementId")
FIELD_ENT_DISPLAY_NAME = os.environ.get("FIELD_ENT_DISPLAY_NAME", "displayName")
FIELD_ENT_APP_ID = os.environ.get("FIELD_ENT_APP_ID", "appId")
FIELD_ENT_APP_NAME = os.environ.get("FIELD_ENT_APP_NAME", "appName")
FIELD_ENT_TYPE = os.environ.get("FIELD_ENT_TYPE", "entitlementType")
FIELD_ENT_CRITICALITY = os.environ.get("FIELD_ENT_CRITICALITY", "criticality")

# --- Server ---
PIPELINE_PORT = int(os.environ.get("PIPELINE_PORT", 8001))
