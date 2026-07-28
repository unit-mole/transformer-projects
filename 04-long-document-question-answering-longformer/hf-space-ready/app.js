import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm';

const MODEL_ID = 'Xenova/distilbert-base-cased-distilled-squad';
env.allowLocalModels = false;
env.useBrowserCache = true;

const $ = (id) => document.getElementById(id);
const state = { sourceName: 'pasted-text', qa: null, runtime: null, lastResult: null };

function words(text){return (text.trim().match(/\S+/g)||[])}
function normalize(text){return text.replace(/\r\n?/g,'\n').replace(/[ \t]+/g,' ').replace(/\n{3,}/g,'\n\n').trim()}
function escapeHtml(value){return value.replace(/[&<>'"]/g,(c)=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
function tokens(text){return new Set((text.toLowerCase().match(/[a-z0-9]+/g)||[]).filter((x)=>x.length>2))}

function setStatus(message,kind='info'){$('statusMessage').textContent=message;$('statusMessage').className=`status ${kind}`}
function setProgress(prefix,value,label){$(prefix+'Progress').value=Math.max(0,Math.min(100,value||0));$(prefix+'ProgressText').textContent=label}
function updateStats(){const text=normalize($('documentText').value);$('inputCharacters').textContent=text.length.toLocaleString();$('inputWords').textContent=words(text).length.toLocaleString();$('sourceName').textContent=state.sourceName}

function chunkDocument(text,chunkWords,overlapWords){
  const matches=[...text.matchAll(/\S+/g)]; const result=[]; let start=0; let id=0;
  while(start<matches.length){const end=Math.min(start+chunkWords,matches.length);const startChar=matches[start].index;const endMatch=matches[end-1];const endChar=endMatch.index+endMatch[0].length;result.push({id:id++,text:text.slice(startChar,endChar),startChar,endChar});if(end===matches.length)break;start=Math.max(start+1,end-overlapWords)}
  return result;
}
function retrievalScore(question,chunk){const q=tokens(question),c=tokens(chunk);if(!q.size)return 0;let hit=0;for(const token of q)if(c.has(token))hit++;return hit/q.size}
function supportingParagraph(chunkText,answer){const ps=chunkText.split(/\n\s*\n/).map((x)=>x.trim()).filter(Boolean);const target=answer.toLowerCase();return ps.find((p)=>p.toLowerCase().includes(target))||ps.sort((a,b)=>retrievalScore(answer,b)-retrievalScore(answer,a))[0]||chunkText}
function highlightEvidence(paragraph,answer){const lower=paragraph.toLowerCase(),needle=answer.toLowerCase();const i=lower.indexOf(needle);if(i<0)return escapeHtml(paragraph);return `${escapeHtml(paragraph.slice(0,i))}<mark class="highlight">${escapeHtml(paragraph.slice(i,i+answer.length))}</mark>${escapeHtml(paragraph.slice(i+answer.length))}`}

async function loadModel(){
  const runtime=$('runtimeSelect').value;
  if(state.qa&&state.runtime===runtime)return state.qa;
  $('loadModelButton').disabled=true;setStatus('Downloading and initializing the browser QA model. The first load can take a few minutes.');setProgress('model',4,'Starting model load');
  try{
    const device=runtime==='webgpu'?'webgpu':'wasm';
    state.qa=await pipeline('question-answering',MODEL_ID,{device,progress_callback:(p)=>{const pct=Number.isFinite(p.progress)?p.progress:35;setProgress('model',pct,p.status||p.file||'Loading model')}});
    state.runtime=runtime;setProgress('model',100,'Model ready');setStatus('Browser QA model loaded and cached for this session.','success');return state.qa;
  }catch(error){state.qa=null;setStatus(`Model loading failed: ${error.message}`,'error');throw error}finally{$('loadModelButton').disabled=false}
}

async function answerDocument(){
  const question=normalize($('question').value),documentText=normalize($('documentText').value);if(question.length<3)throw new Error('Enter a focused question.');if(documentText.length<30)throw new Error('Provide a longer document before asking a question.');if(documentText.length>1_000_000)throw new Error('Document exceeds the 1,000,000-character browser limit.');
  const chunkWords=Number($('chunkWords').value),overlap=Number($('overlapWords').value),candidateCount=Number($('candidateChunks').value);if(overlap>=chunkWords)throw new Error('Chunk overlap must be smaller than chunk size.');
  const chunks=chunkDocument(documentText,chunkWords,overlap).map((c)=>({...c,retrievalScore:retrievalScore(question,c.text)})).sort((a,b)=>b.retrievalScore-a.retrievalScore);const candidates=chunks.slice(0,Math.min(candidateCount,chunks.length));
  const qa=await loadModel();const started=performance.now();const results=[];setProgress('inference',1,'Evaluating candidate chunks');
  for(let i=0;i<candidates.length;i++){
    const chunk=candidates[i];const raw=await qa(question,chunk.text);const answer=Array.isArray(raw)?raw[0]:raw;if(answer?.answer){results.push({...answer,chunkId:chunk.id,chunkText:chunk.text,retrievalScore:chunk.retrievalScore,combinedScore:Number(answer.score||0)*(0.75+0.25*chunk.retrievalScore)})}setProgress('inference',Math.round(((i+1)/candidates.length)*100),`Evaluated ${i+1} of ${candidates.length}`)
  }
  if(!results.length)throw new Error('The model did not return a usable answer. Try a more focused question.');
  results.sort((a,b)=>b.combinedScore-a.combinedScore);const best=results[0];const paragraph=supportingParagraph(best.chunkText,best.answer);const latency=(performance.now()-started)/1000;
  return {answer:best.answer,confidenceProxy:Number(best.score||0),supportingParagraph:paragraph,highlightedEvidenceHtml:highlightEvidence(paragraph,best.answer),totalChunks:chunks.length,evaluatedChunks:candidates.length,latencySeconds:latency,runtime:$('runtimeSelect').value,browserModelId:MODEL_ID,corePythonModelId:'anmol-unitmole/longformer-qasper-document-qa',candidateResults:results.slice(0,6).map((r)=>({answer:r.answer,confidence_proxy:Number(r.score||0),retrieval_score:r.retrievalScore,chunk_id:r.chunkId}))};
}

function render(result){$('answerOutput').textContent=result.answer;$('confidenceOutput').textContent=result.confidenceProxy.toFixed(6);$('supportingOutput').textContent=result.supportingParagraph;$('evidenceOutput').innerHTML=result.highlightedEvidenceHtml;$('metricChunks').textContent=result.totalChunks;$('metricEvaluated').textContent=result.evaluatedChunks;$('metricLatency').textContent=`${result.latencySeconds.toFixed(2)} s`;$('metricRuntime').textContent=result.runtime.toUpperCase();$('diagnosticsOutput').textContent=JSON.stringify({source_name:state.sourceName,browser_model:result.browserModelId,evaluated_python_model:result.corePythonModelId,total_chunks:result.totalChunks,evaluated_candidate_chunks:result.evaluatedChunks,latency_seconds:Number(result.latencySeconds.toFixed(4)),top_candidates:result.candidateResults},null,2);$('exportButton').disabled=false}
function reset(){state.lastResult=null;$('answerOutput').textContent='No answer generated yet.';$('confidenceOutput').textContent='—';$('supportingOutput').textContent='No supporting paragraph selected yet.';$('evidenceOutput').textContent='Highlighted evidence will appear here.';$('diagnosticsOutput').textContent='{}';['metricChunks','metricEvaluated','metricLatency','metricRuntime'].forEach((id)=>$(id).textContent='—');$('exportButton').disabled=true;setProgress('inference',0,'Not started')}

async function loadSamples(){const response=await fetch('./samples/index.json');const data=await response.json();for(const sample of data.samples){const option=document.createElement('option');option.value=sample.file;option.textContent=sample.name;option.dataset.question=sample.question;$('sampleSelect').appendChild(option)}}
async function selectSample(){const option=$('sampleSelect').selectedOptions[0];if(!option?.value)return;const response=await fetch(`./samples/${encodeURIComponent(option.value)}`);$('documentText').value=await response.text();$('question').value=option.dataset.question||'';state.sourceName=option.value;updateStats();setStatus(`Loaded sample: ${option.textContent}`,'success')}
async function readFile(){const file=$('fileInput').files?.[0];if(!file)return;const ext=file.name.split('.').pop().toLowerCase();if(!['txt','md','csv'].includes(ext))throw new Error('This credit-free browser demo supports TXT, Markdown, and CSV files.');let text=await file.text();if(ext==='csv'){const lines=text.split(/\r?\n/).filter(Boolean);text=lines.map((line)=>line.split(',').join(' ')).join('\n')} $('documentText').value=text;state.sourceName=file.name;$('sampleSelect').value='';updateStats();setStatus(`Loaded ${file.name}. Review the extracted text before asking a question.`,'success')}
function exportResult(){if(!state.lastResult)return;const payload={exported_at:new Date().toISOString(),source_name:state.sourceName,question:$('question').value,...state.lastResult,highlightedEvidenceHtml:undefined};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='project04-long-document-qa-result.json';a.click();URL.revokeObjectURL(url)}

$('documentText').addEventListener('input',()=>{state.sourceName='pasted-text';updateStats()});$('sampleSelect').addEventListener('change',()=>selectSample().catch((e)=>setStatus(e.message,'error')));$('fileInput').addEventListener('change',()=>readFile().catch((e)=>setStatus(e.message,'error')));$('loadModelButton').addEventListener('click',()=>loadModel());$('askButton').addEventListener('click',async()=>{$('askButton').disabled=true;setStatus('Preparing document chunks and running browser inference.');try{const result=await answerDocument();state.lastResult=result;render(result);setStatus('Answer generated. Review the highlighted evidence before relying on it.','success')}catch(e){setStatus(e.message,'error')}finally{$('askButton').disabled=false}});$('resetButton').addEventListener('click',reset);$('exportButton').addEventListener('click',exportResult);
if(!('gpu'in navigator)){const option=$('runtimeSelect').querySelector('option[value="webgpu"]');option.disabled=true;option.textContent='WebGPU (not available)'}
reset();updateStats();loadSamples().catch((e)=>setStatus(`Sample loading failed: ${e.message}`,'error'));
