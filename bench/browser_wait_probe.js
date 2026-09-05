// Real browser timers, AudioWorklet and stop button; mocked socket/silent input.
async (page) => {
  const results=[];
  for (const domain of ['clinic','pizza']) {
    await page.setViewportSize({width:domain==='clinic'?390:1100,height:900});
    await page.goto(`http://127.0.0.1:8000/?domain=${domain}&mode=demo`);
    await page.evaluate(() => {
      const probe=window.__waitProbe={controls:[],frames:0};
      navigator.mediaDevices.getUserMedia=async()=>{
        probe.inputContext=new AudioContext({sampleRate:16000});
        const destination=probe.inputContext.createMediaStreamDestination();
        probe.source=probe.inputContext.createConstantSource(); probe.source.offset.value=0;
        probe.source.connect(destination); probe.source.start(); await probe.inputContext.resume();
        return probe.stream=destination.stream;
      };
      window.WebSocket=class {
        static OPEN=1;
        constructor(){this.readyState=1;this.bufferedAmount=0;probe.socket=this;queueMicrotask(()=>this.deliver({type:'ready'}));}
        deliver(value){this.onmessage?.({data:JSON.stringify(value)});}
        send(value){
          if(value instanceof ArrayBuffer){probe.frames++;return;}
          const control=JSON.parse(value);probe.controls.push(control);
          if(control.type==='start') queueMicrotask(()=>this.deliver({type:'started',session_id:'wait-fixture'}));
          if(control.type==='stop') queueMicrotask(()=>this.deliver({type:'done',status:'interrupted',slots:{},session_id:'wait-fixture'}));
        }
        close(){this.readyState=3;}
      };
    });
    await page.getByRole('button',{name:'Start conversation',exact:true}).click();
    await page.waitForFunction(()=>current?.phase==='recording' && __waitProbe.frames>2);
    const immediate=await page.evaluate(()=>{
      __waitProbe.socket.deliver({type:'state',state:'PROCESSING',slots:{}});
      __waitProbe.socket.deliver({type:'progress',progress_id:'asr',stage:'transcribing',active:true});
      return {status:document.getElementById('status').textContent,elapsedHidden:document.getElementById('progress-time').hidden};
    });
    if(immediate.status!=='Transcribing'||!immediate.elapsedHidden) throw new Error(JSON.stringify(immediate));
    await page.waitForFunction(()=>document.getElementById('message').textContent.includes('Still waiting for the transcription'),{},{timeout:12000});
    const waiting=await page.evaluate(()=>{
      __waitProbe.socket.deliver({type:'progress',progress_id:'obsolete',stage:'transcribing',active:false});
      return {status:document.getElementById('status').textContent,message:document.getElementById('message').textContent,
        elapsed:document.getElementById('progress-time').textContent,endEnabled:!document.getElementById('record').disabled,
        elapsedExcludedFromLiveRegion:document.getElementById('progress-time').getAttribute('aria-hidden')==='true',
        noHorizontalOverflow:document.documentElement.scrollWidth<=innerWidth};
    });
    if(waiting.status!=='Transcribing'||!waiting.endEnabled||!waiting.noHorizontalOverflow||!waiting.elapsedExcludedFromLiveRegion) throw new Error(JSON.stringify(waiting));
    await page.screenshot({path:`output/playwright/waiting_${domain}.png`,fullPage:true});
    const restored=await page.evaluate(()=>{
      __waitProbe.socket.deliver({type:'state',state:'LISTENING',slots:{}});
      const stillTranscribing=document.getElementById('status').textContent==='Transcribing';
      __waitProbe.socket.deliver({type:'progress',progress_id:'asr',stage:'transcribing',active:false});
      const restored=document.getElementById('status').textContent==='Listening' && document.getElementById('progress-time').hidden;
      __waitProbe.socket.deliver({type:'progress',progress_id:'tts',stage:'synthesizing',active:true});
      return {stillTranscribing,restored,preparing:document.getElementById('status').textContent==='Preparing reply'};
    });
    if(!Object.values(restored).every(Boolean)) throw new Error(JSON.stringify(restored));
    await page.getByRole('button',{name:'End session',exact:true}).click();
    await page.waitForFunction(()=>current===null && __waitProbe.controls.some(control=>control.type==='stop'));
    const stopped=await page.evaluate(async()=>{
      __waitProbe.socket.deliver({type:'progress',progress_id:'late',stage:'synthesizing',active:true});
      __waitProbe.socket.deliver({type:'error',message:'Late fixture result must be ignored.'});
      await new Promise(resolve=>setTimeout(resolve,350));
      const result={status:document.getElementById('status').textContent,elapsedHidden:document.getElementById('progress-time').hidden,
        inputEnded:__waitProbe.stream.getTracks().every(track=>track.readyState==='ended'),stopCount:__waitProbe.controls.filter(control=>control.type==='stop').length};
      __waitProbe.source.stop();await __waitProbe.inputContext.close();
      return result;
    });
    if(stopped.status!=='Session ended'||!stopped.elapsedHidden||!stopped.inputEnded||stopped.stopCount!==1) throw new Error(JSON.stringify(stopped));
    results.push({domain,immediate,waiting,restored,stopped});
  }
  const report={kind:'browser_slow_turn_diagnostic',provider_requests:0,input:'synthetic silent stream',transport:'mock WebSocket',
    timing:'real browser timers, >=8 seconds per domain',limitations:'No physical microphone or provider latency measurement.',results};
  await page.evaluate(value=>{window.__waitProbeReport=value;},report);
  return report;
}
