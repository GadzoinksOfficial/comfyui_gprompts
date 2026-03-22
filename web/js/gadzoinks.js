import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { getPngMetadata, getWebpMetadata, importA1111, getLatentMetadata } from "../../scripts/pnginfo.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { createElement as $el, getClosestOrSelf, setAttributes } from "./utils_dom.js";

console.log("LOADED GPROMPTS");

function dprint(...args) {
   // console.log(...args);
}


app.registerExtension({
    name: "Comfy.GPrompts.settings",
    settings: [
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
            id: "Gadzoinks.immich_save_also",
            name: "Save Image to Disk",
            defaultValue: "",
            type: "boolean",
            options: [
                { value: true, text: "On" },
                { value: false, text: "Off" },
            ],
            category: ["Gadzoinks", "Account", "Save_also"],
            async onChange(value) { setbackendVariables({immich_save_also: value}); }
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
            id: "Gadzoinks.immich.hostname",
            name: "Immich server hostname",
            type: "text",
            defaultValue: "example.local",
            category: ["Gadzoinks", "Account", "Hostname"],
            async onChange(value) { setbackendVariables({immich_hostname: value}); }
        },
        {
            id: "Gadzoinks.immich.apikey",
            name: "Immich Server Api Key",
            type: "text",
            defaultValue: "spoon",
            category: ["Gadzoinks", "Account", "Apikey"],
            async onChange(value) { setbackendVariables({immich_apikey: value}); }
        },
    ],
	async setup() {
		dprint("Setting up Gadzoinks extension setup() foo");
		// Force-sync all settings to backend on load
		setTimeout(async () => {
			const settingMap = {
			    "Gadzoinks.immich.base_tags":    "immich_base_tags",
			    "Gadzoinks.immich.default_album": "immich_default_album",
			    "Gadzoinks.immich_save_also":     "immich_save_also",
			    "Gadzoinks.immich.port":          "immich_port",
			    "Gadzoinks.immich.hostname":      "immich_hostname",
			    "Gadzoinks.immich.apikey":        "immich_apikey",
			};

			const payload = {};
			for (const [settingId, backendKey] of Object.entries(settingMap)) {
			    payload[backendKey] = app.ui.settings.getSettingValue(settingId);
			}
			dprint("Syncing settings to backend:", payload);
			await setbackendVariables(payload);
		}, 500);
		api.addEventListener("gprompts_executed", ({detail}) => {
		 	const {node_id, result} = detail;
			dprint("gprompts_executed result",result);
			dprint("gprompts_executed node_id",node_id);
			if (!node_id || !result) { dprint("gprompts_executed node_id return B1"); return; }
			dprint("gprompts_executed node_id A1");
			const node = app.graph.getNodeById(node_id);
			if (!node) { dprint("no node"); return;}
			if (!node.properties._meta) { node.properties._meta={}; }
			dprint("gprompts_executed node_id A2 node",node);
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
		api.addEventListener("gadzoinks.request_settings", async () => {
			dprint("Backend requested settings, sending...");
			const settingMap = {
			    "Gadzoinks.immich.base_tags":     "immich_base_tags",
			    "Gadzoinks.immich.default_album":  "immich_default_album",
			    "Gadzoinks.immich_save_also":      "immich_save_also",
			    "Gadzoinks.immich.port":           "immich_port",
			    "Gadzoinks.immich.hostname":       "immich_hostname",
			    "Gadzoinks.immich.apikey":         "immich_apikey",
			};
			const payload = {};
			for (const [settingId, backendKey] of Object.entries(settingMap)) {
			    payload[backendKey] = app.ui.settings.getSettingValue(settingId);
			}
			dprint("Syncing settings to backend:", payload);
			await setbackendVariables(payload);
		});
	},    
   async init(app) {
	dprint("Setting up Gadzoinks init() foo");
   },
});   
async function setbackendVariables(params = {}) {
    const urlParams = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
        if (value != null) {  urlParams.append(key, value); }
    }
    // Only make the API call if we have at least one parameter
    if (urlParams.toString()) {
        const response = await api.fetchApi(`/gprompts/setting?${urlParams.toString()}`);
        return response;
    }
    return null;
}

