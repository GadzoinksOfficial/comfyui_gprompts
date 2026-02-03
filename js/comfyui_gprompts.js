import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

function dprint(...args) {
	// console.log(...args);
}

app.registerExtension( {
	name: "gprompts.settings",
	settings: [],
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
	    dprint("beforeRegisterNodeDef",nodeData);
	    try {
		const response = await api.fetchApi(`/gprompts/prompt`);
		prompt =  response.prompt;
	    } catch (error) {
		console.error('Error:', error);
		return null;
	    }	
	},
	async afterConfigureGraph(nodeType, nodeData, app) {
		dprint("afterConfigureGraph",nodeData);
	},
	async beforeProjectSave(nodeType, nodeData, app) {
		dprint("beforeProjectSave",nodeData);
	},
	async setup() {
		dprint("setup");
		 api.addEventListener("promptQueued", ({detail}) => {
			 const {node_id, result} = detail;
			 dprint("promptQueued detail",detail);
			 dprint("promptQueued result",result);
			 dprint("promptQueued node_id",node_id);
		 });

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
	},
	async nodeCreated(node) {
		dprint("nodeCreated");
		dprint(node);
		dprint(node.type);
		if (node.type !== "GPrompts") {
			return;
		}
		try {
			const response = await api.fetchApi(`/gprompts/prompt`);
			const processedResult = response.prompt;
			if (processedResult) {
				if (!node.properties._meta) {
					node.properties._meta = {};
			    	}
			    	node.properties._meta.computed_result = processedResult;
			    for (const w of node.widgets || []) {
				if (w.name === "computed_prompt") {
				    w.value = processedResult;
				    dprint("nodeCreated",w);
				    break;
				}
			    }
			    app.graph.setDirtyCanvas(true);
			}
		} catch (error) {
			console.error("Error fetching computed prompt:", error);
		}
	},
});

