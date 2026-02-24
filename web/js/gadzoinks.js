import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { getPngMetadata, getWebpMetadata, importA1111, getLatentMetadata } from "../../scripts/pnginfo.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { createElement as $el, getClosestOrSelf, setAttributes } from "./utils_dom.js";

console.log("LOADED GPROMPTS");

function dprint(...args) {
   console.log(...args);
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
        try {
	}
	catch(exception) {
            dprint("ComfyUI is outdated. New style menu based features are disabled.");
        }
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

