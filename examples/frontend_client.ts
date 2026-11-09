type Status = 'PASS'|'FAIL'|'UNVERIFIED';
type Candidate = {id:string;status:Status;validation_scope:string;executable:false;checks:Record<string,{status:Status}>;camera_from_tcp:number[][]};
export async function plan(base:string,update:(s:unknown)=>void):Promise<Candidate[]> {
 const response=await fetch(`${base}/v3/plan`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_id:'0000',planner:{bbox_scale:1.25},grippers:['2fg7','rg2','rg6'].map(model=>({model,name:`OnRobot_${model.toUpperCase()}`}))})});
 if(!response.ok)throw new Error(await response.text());const job=await response.json();
 for(;;){const response=await fetch(`${base}${job.status_url}`);if(!response.ok)throw new Error(await response.text());const state=await response.json();update(state);if(state.status==='failed')throw new Error(state.error);if(state.status==='completed')break;await new Promise(resolve=>setTimeout(resolve,2000));}
 const result=await fetch(`${base}/v3/jobs/${job.job_id}/files/top10_grasps.json`);if(!result.ok)throw new Error(await result.text());return result.json();
}
