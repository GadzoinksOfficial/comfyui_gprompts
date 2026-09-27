import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { getPngMetadata, getWebpMetadata, importA1111, getLatentMetadata } from "../../scripts/pnginfo.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { createElement as $el, getClosestOrSelf, setAttributes } from "./utils_dom.js";

console.log("LOADED GPROMPTS");

function dprint(...args) {
   // console.log(...args);
}

// Single source of truth for the settingId -> backend key mapping.
// (Previously duplicated in setup() and the request_settings listener.)
const SETTING_MAP = {
    "Gadzoinks.immich.base_tags":        "immich_base_tags",
    "Gadzoinks.immich.default_album":    "immich_default_album",
    "Gadzoinks.immich_save_also":        "immich_save_also",
    "Gadzoinks.immich.port":             "immich_port",
    "Gadzoinks.immich.hostname":         "immich_hostname",
    "Gadzoinks.immich.apikey":           "immich_apikey",
    "Gadzoinks.immich.include.hostname": "immich_includehostname",
    "Gadzoinks.llm.api_key":             "llm_api_key",
    "Gadzoinks.llm.api_keys":            "llm_api_keys",
    "Gadzoinks.debug_logging":           "debug_logging",
};

function collectSettings() {
    const payload = {};
    for (const [settingId, backendKey] of Object.entries(SETTING_MAP)) {
        const value = app.ui.settings.getSettingValue(settingId);
        // Skip undefined/null so an uninitialized settings store sends
        // nothing rather than blanks. The backend also guards against
        // blanks/placeholders, but don't send junk in the first place.
        if (value !== undefined && value !== null) {
            payload[backendKey] = value;
        }
    }
    return payload;
}

async function syncSettingsToBackend(reason) {
    const payload = collectSettings();
    dprint(`Syncing settings to backend (${reason}):`, payload);
    return setbackendVariables(payload);
}

app.registerExtension({
    name: "Comfy.GPrompts.settings",
    settings: [
        {
            id: "Gadzoinks.immich.include.hostname",
            name: "Include hostname in metadata",
            defaultValue: true,
            type: "boolean",
            options: [
                { value: true, text: "On" },
                { value: false, text: "Off" },
            ],
            category: ["Gadzoinks", "Account", "IncludeHostname"],
            async onChange(value) { setbackendVariables({immich_includehostname: value}); }
        },
        {
            id: "Gadzoinks.immich_save_also",
            name: "Save Image to Disk",
            defaultValue: true,
            type: "boolean",
            options: [
                { value: true, text: "On" },
                { value: false, text: "Off" },
            ],
            category: ["Gadzoinks", "Account", "Save_also"],
            async onChange(value) { setbackendVariables({immich_save_also: value}); }
        },
        {
            id: "Gadzoinks.immich.base_tags",
            name: "Default Tags (optional)",
            type: "text",
            defaultValue: "",
            category: ["Gadzoinks", "Account", "Tags"],
            async onChange(value) { setbackendVariables({immich_base_tags: value}); }
        },
        {
            id: "Gadzoinks.immich.default_album",
            name: "Default Album (optional)",
            type: "text",
            defaultValue: "",
            category: ["Gadzoinks", "Account", "Album"],
            async onChange(value) { setbackendVariables({immich_default_album: value}); }
        },
        {
            id: "Gadzoinks.immich.port",
            name: "Immich server port",
            type: "text",
            defaultValue: "2283",
            category: ["Gadzoinks", "Account", "Port"],
            async onChange(value) { setbackendVariables({immich_port: value}); }
        },
        {
            // NOTE: defaults are now empty instead of "example.local"/"secret".
            // A fresh client answering the settings broadcast with those
            // placeholder values would overwrite real settings on the server
            // (the backend also rejects the old placeholders defensively).
            id: "Gadzoinks.immich.hostname",
            name: "Immich server hostname",
            type: "text",
            defaultValue: "",
            category: ["Gadzoinks", "Account", "Hostname"],
            async onChange(value) { setbackendVariables({immich_hostname: value}); }
        },
        {
            id: "Gadzoinks.immich.apikey",
            name: "Immich Server Api Key",
            type: "text",
            defaultValue: "",
            category: ["Gadzoinks", "Account", "Apikey"],
            async onChange(value) { setbackendVariables({immich_apikey: value}); }
        },
        {
            // API keys for "Prompt Enhancer Loader (API)". Kept here, not on the
            // node, so they never end up in saved workflows or image metadata.
            id: "Gadzoinks.llm.api_key",
            name: "LLM API key (default)",
            tooltip: "Used by Prompt Enhancer Loader (API) when its api_key_name is blank.",
            type: "text",
            defaultValue: "",
            category: ["Gadzoinks", "LLM", "ApiKey"],
            async onChange(value) { setbackendVariables({llm_api_key: value}); }
        },
        {
            id: "Gadzoinks.llm.api_keys",
            name: "LLM API keys, named (name=key; name=key)",
            tooltip: "Extra keys for other providers, e.g. openrouter=sk-or-...; dashscope=sk-... " +
                     "Select one on the loader with api_key_name.",
            type: "text",
            defaultValue: "",
            category: ["Gadzoinks", "LLM", "ApiKeys"],
            async onChange(value) { setbackendVariables({llm_api_keys: value}); }
        },
        {
            // Debug detail in the ComfyUI console. Turn on when reporting a problem.
            id: "Gadzoinks.debug_logging",
            name: "Debug logging (console)",
            tooltip: "Show Gadzoinks debug messages in the ComfyUI console and log. " +
                     "Turn on when reporting a problem; API keys are never logged.",
            type: "boolean",
            defaultValue: false,
            category: ["Gadzoinks", "Debug", "Logging"],
            async onChange(value) { setbackendVariables({debug_logging: value}); }
        },
    ],
    async setup() {
        dprint("Setting up Gadzoinks extension setup()");
        // Force-sync all settings to backend on load
        setTimeout(() => syncSettingsToBackend("initial load"), 500);

        // Re-sync whenever the websocket reconnects. Over a weak link (VPN)
        // the client semi-disconnects; on reconnect the server may have
        // restarted or missed a request_settings round-trip, so push fresh
        // values without being asked.
        api.addEventListener("reconnected", () => {
            dprint("Websocket reconnected");
            syncSettingsToBackend("reconnect");
        });

        api.addEventListener("gprompts_executed", ({detail}) => {
            const {node_id, result} = detail;
            dprint("gprompts_executed result", result);
            dprint("gprompts_executed node_id", node_id);
            if (!node_id || !result) { dprint("gprompts_executed node_id return B1"); return; }
            const node = app.graph.getNodeById(node_id);
            if (!node) { dprint("no node"); return; }
            if (!node.properties._meta) { node.properties._meta = {}; }
            node.properties._meta.computed_result = result;
            node.properties._meta.computed_prompt = result;
            for (const widget of node.widgets || []) {
                if (widget.name === "computed_prompt") {
                    widget.value = result;
                    dprint("Updated widget value");
                    break;
                }
            }
            app.graph.setDirtyCanvas(true);
            dprint(node.properties._meta);
        });

        api.addEventListener("gadzoinks.request_settings", () => {
            dprint("Backend requested settings, sending...");
            syncSettingsToBackend("backend request");
        });
    },
    async init(app) {
        dprint("Setting up Gadzoinks init()");
    },
});

async function setbackendVariables(params = {}) {
    const payload = {};
    for (const [key, value] of Object.entries(params)) {
        if (value != null) { payload[key] = value; }
    }
    if (Object.keys(payload).length === 0) { return null; }
    try {
        // POST with a JSON body so the API key doesn't end up in URLs /
        // server access logs. Falls back to the legacy GET endpoint if the
        // backend is an older version without the POST route.
        const response = await api.fetchApi("/gprompts/setting", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (response && response.status === 404) {
            return legacyGetFallback(payload);
        }
        return response;
    } catch (e) {
        dprint("POST /gprompts/setting failed, trying legacy GET:", e);
        return legacyGetFallback(payload);
    }
}

async function legacyGetFallback(payload) {
    const urlParams = new URLSearchParams();
    for (const [key, value] of Object.entries(payload)) {
        urlParams.append(key, value);
    }
    return api.fetchApi(`/gprompts/setting?${urlParams.toString()}`);
}

