async (page) => {
  // Engineering fixture only: real DOM rendering, mocked configuration, microphone and transport.
  // Run in a fresh named Playwright CLI context; this installs a page initialization script.
  const configuration = {
    domain_id: 'collection_fixture', domain_title: 'Equipment reservation fixture',
    demo_supported: true, demo_state_kind: 'configured_collection_scoped', demo_issues: [],
    demo_collection: {name:'reservations',label:'Reservation',root_slots:['return_date'],item_slots:['asset','quantity'],min_items:1,max_items:10},
    demo_labels: {reservations:'Reservations',asset:'Equipment',quantity:'Quantity',return_date:'Return date'},
    demo_values: {camera:'Camera',tripod:'Tripod',light:'Light'},
    demo_ui: {title:'Reserve equipment · browser fixture',note:'Mocked interface check. No reservation is made.',state_title:'Saved reservations',state_empty:'Your equipment will appear here.',completion_text:'Fixture completed. No reservation was made.'},
    domains: [{id:'collection_fixture',title:'Equipment reservation fixture'}],
  };
  await page.route('http://127.0.0.1:8000/?domain=clinic&mode=demo', async route => {
    const response = await route.fetch();
    let html = await response.text();
    html = html.replace(/(<script id="capture-config"[^>]*>)([\s\S]*?)(<\/script>)/, (_, before, value, after) =>
      before + JSON.stringify({...JSON.parse(value),...configuration}) + after);
    await route.fulfill({response,body:html});
  });
  await page.addInitScript(() => {
    const fixture = window.__collectionFixture = {sockets:[],tracks:[]};
    window.AudioContext = class {
      constructor() {this.sampleRate=16000;this.state='running';this.destination={};this.audioWorklet={addModule:async()=>{}};}
      async resume() {} async close() {this.state='closed';}
      createMediaStreamSource() {return {connect(){},disconnect(){}};}
    };
    window.AudioWorkletNode = class {
      constructor() {this.port={close(){},onmessage:null,postMessage:value=>{
        if(value.type==='stop') setTimeout(()=>this.port.onmessage?.({data:{type:'finished',samples:0}}),0);
      }};}
      connect() {} disconnect() {}
    };
    Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:async()=>{
      const track={stopped:false,onended:null,stop(){this.stopped=true;}};
      fixture.tracks.push(track);
      return {getTracks:()=>[track],getAudioTracks:()=>[track]};
    }});
    window.WebSocket = class {
      static OPEN=1;
      constructor(){this.readyState=1;this.bufferedAmount=0;fixture.sockets.push(this);setTimeout(()=>this.emit({type:'ready'}),0);}
      emit(value){this.onmessage?.({data:JSON.stringify(value)});}
      send(data){const value=JSON.parse(data);
        if(value.type==='start') setTimeout(()=>{
          this.emit({type:'started',session_id:'collection-fixture'});
          this.emit({type:'state',state:'LISTENING',slots:{reservations:[],return_date:'2026-09-20'}});
        },0);
        if(value.type==='stop') setTimeout(()=>this.emit({type:'done',status:'interrupted',slots:{}}),0);
      }
      close(){this.readyState=3;}
    };
  });
  await page.goto('http://127.0.0.1:8000/?domain=clinic&mode=demo');
  await page.locator('#record').click();
  await page.waitForFunction(()=>window.__collectionFixture.sockets.length===1 && document.querySelector('#status').textContent==='Listening');
  await page.evaluate(()=>{
    const slots={reservations:[{id:2,asset:'camera',quantity:4},{id:3,asset:'tripod',quantity:7}],return_date:'2026-09-20'};
    const proposal={state:{reservations:[{id:2,asset:'light',quantity:4},{id:3,asset:'tripod',quantity:7}],return_date:'2026-09-20'},
      remaining_addresses:[{item_id:2,slot:'quantity'}]};
    window.__collectionFixture.sockets[0].emit({type:'state',state:'LISTENING',slots,pending_proposal:proposal});
    // Exercise the actual renderer with shared objects too: deleting a preview field must not mutate the source.
    const shared={state:slots,remaining_addresses:[{item_id:2,slot:'quantity'}]};
    showProposal(shared);
    if(shared.state.reservations[0].quantity!==4) throw new Error('Rendering mutated the source row.');
    showProposal(proposal);
  });
  const saved=await page.locator('#slots dd[data-item-id="2"]').innerText();
  const held=await page.locator('#proposed-slots dd[data-item-id="2"]').innerText();
  if(!saved.includes('Camera') || !saved.includes('Quantity: 4')) throw new Error('Saved row changed during partial answer.');
  if(!held.includes('Light') || held.includes('Quantity: 4') || !held.includes('Quantity: not specified')) throw new Error('Draft lost known choice or inherited stale quantity.');
  const other=await page.locator('#proposed-slots dd[data-item-id="3"]').innerText();
  if(!other.includes('Tripod') || !other.includes('Quantity: 7')) throw new Error('Unrelated row was masked.');
  if(!(await page.locator('#proposed-slots').innerText()).includes('2026-09-20')) throw new Error('Root date was masked.');
  const note=await page.locator('#proposal-note').innerText();
  if(!note.includes('Reservation 1: Quantity') || !note.includes('saved details have not changed')) throw new Error('Missing row-specific pending disclosure.');
  for(const [name,width,height] of [['desktop',1365,1100],['mobile',390,844]]) {
    await page.setViewportSize({width,height});
    if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)) throw new Error(`${name} overflow.`);
    await page.screenshot({path:`output/playwright/collection_linked_${name}.png`,fullPage:true});
  }
  await page.evaluate(()=>window.__collectionFixture.sockets[0].emit({type:'state',state:'LISTENING',
    slots:{reservations:[{id:2,asset:'camera',quantity:4},{id:3,asset:'tripod',quantity:7}],return_date:'2026-09-20'},
    pending_proposal:{state:{reservations:[{id:2,asset:'camera',quantity:4},{id:3,asset:'tripod',quantity:7},{id:4,asset:'light',quantity:9}],return_date:'2026-09-20'},remaining_addresses:[{item_id:4,slot:'quantity'}]}}));
  const newRow=await page.locator('#proposed-slots dd[data-item-id="4"]').innerText();
  if(!newRow.includes('Light') || !newRow.includes('Quantity: not specified') || newRow.includes('Quantity: 9')) throw new Error('New row draft shows unresolved quantity.');
  if(await page.locator('#slots dd[data-item-id="4"]').count()) throw new Error('New pending row leaked into saved state.');
  if(!(await page.locator('#proposed-slots dd[data-item-id="2"]').innerText()).includes('Quantity: 4')) throw new Error('New row question masked an existing row.');
  await page.evaluate(()=>window.__collectionFixture.sockets[0].emit({type:'state',state:'CONFIRMING',
    slots:{reservations:[{id:2,asset:'camera',quantity:4},{id:3,asset:'tripod',quantity:7},{id:4,asset:'light',quantity:2}],return_date:'2026-09-20'},pending_proposal:null}));
  if(await page.locator('#proposal-panel').isVisible() || !(await page.locator('#slots dd[data-item-id="4"]').innerText()).includes('Quantity: 2')) throw new Error('Resolved commit did not clear draft or show final quantity.');
  await page.locator('#record').click();
  await page.waitForFunction(()=>window.__collectionFixture.tracks.every(track=>track.stopped));
  return {kind:'real_browser_mocked_configuration_microphone_and_socket',provider_calls:0,checks:['source row objects unmodified','committed old quantity retained','known draft choice retained','only unresolved addressed field masked','unrelated row and root preserved','pending note uses current ordinal','desktop/mobile no overflow','new row unresolved value hidden and uncommitted','final answer updates saved row and clears draft','microphone released']};
}
