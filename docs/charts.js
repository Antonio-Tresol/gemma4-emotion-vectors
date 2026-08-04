const D = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const el = (t,a={})=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
/* the single palette every chart draws from: mirrors the CSS tokens above,
   so a colour can only be changed in one place */
const P = {text:"#0A0A0A", body:"#262626", muted:"#737373", border:"#E5E5E5",
  navy:"#1d3557", teal:"#457b9d", green:"#009E73", amber:"#E69F00", red:"#e63946",
  orange:"#CC785C", greySoft:"#D4D4D4", greyMid:"#BDBDBD", alert:"#b3202c"};
const C = {valence:P.navy, arousal:P.teal, dominance:P.green, length:P.amber, evr:P.greySoft};

/* Inline glossary, the same idea as the <Term> component in the refusal
   write-up: a dotted underline on a technical word, one sentence of plain
   English on hover, and a click that lands on the full entry.

   The definitions are NOT held here. Each one lives in its <details> in
   section 11 (content/12-glossary.html), and the hover reads the short line
   back out of that markup, so the page holds one copy of every definition and
   editing the glossary section edits the tooltips with it. The entry carries
   the longer version and the reference the definition comes from. */
function wireGlossary(){
  const entries = {};
  document.querySelectorAll(".gterm[data-slug]").forEach(entry=>{
    entries[entry.dataset.slug] = {
      id: entry.id,
      term: entry.querySelector(".gt").textContent.trim(),
      // the markup hard-wraps, so collapse before the sentence reaches a tooltip
      short: entry.querySelector(".gs").textContent.replace(/\s+/g," ").trim()};
  });
  document.querySelectorAll("[data-term]").forEach(node=>{
    const key=node.dataset.term, entry=entries[key];
    if(!entry){ console.warn("no glossary entry for", key); return; }  // fail loudly
    // The word becomes a link to its entry. Built here rather than written into
    // the content files, so a term stays a plain <span data-term="..."> to
    // whoever edits the prose and no href can drift from the entry it names.
    const link=document.createElement("a");
    link.className="term"; link.href="#"+entry.id;
    while(node.firstChild) link.appendChild(node.firstChild);
    node.appendChild(link);
    // Open on click too, not only on hashchange. A reader who shut the entry by
    // hand and then clicked the same term again would otherwise get nothing: the
    // hash has not changed, so no hashchange fires. This also covers the copies
    // of these terms that the expand-figure view clones out of a how-to block.
    link.addEventListener("click",()=>{
      const opened=document.getElementById(entry.id);
      if(opened) opened.open=true;
    });
    tipOn(link, `<b>${entry.term}</b><span class="t-sub">${entry.short}</span>`
      + `<span class="t-go">See the full entry and its source &rarr;</span>`);
    link.style.cursor="help";
  });
}

/* A link into the glossary has to arrive with the entry already open, whether
   it was clicked on this page or pasted cold into the address bar. CSS :target
   can style a <details> but cannot open one, so the opening happens here.

   Read from location.hash rather than matching :target, and listen for load as
   well as hashchange. A fragment applied after this script runs left the entry
   shut when only :target was consulted, which is a page that looks like the
   link did nothing. */
function openTargetedEntry(){
  const id=decodeURIComponent(location.hash.slice(1));
  const entry=id && document.getElementById(id);
  if(entry && entry.classList.contains("gterm")) entry.open=true;
}
addEventListener("hashchange",openTargetedEntry);
addEventListener("load",openTargetedEntry);

/* one floating tooltip shared by every chart on the page */
const TIP=document.getElementById("tip");
function tipOn(node, html){
  node.style.cursor="default";
  node.addEventListener("mousemove",e=>{
    TIP.innerHTML=typeof html==="function"?html():html;
    TIP.style.opacity=1;
    const r=TIP.getBoundingClientRect();
    let x=e.clientX+14, y=e.clientY+14;
    if(x+r.width>innerWidth-8) x=e.clientX-r.width-14;
    if(y+r.height>innerHeight-8) y=e.clientY-r.height-14;
    TIP.style.left=x+"px"; TIP.style.top=y+"px";
  });
  node.addEventListener("mouseleave",()=>{TIP.style.opacity=0;});
}
const pct=v=>(v*100).toFixed(0)+"%";

/* ---------- method schematic: how a vector is built and read back ---------- */
/* Drawn rather than described. The pipeline is four steps and a reader of this
   page knows all four; a picture states them in the space a paragraph would
   spend re-explaining what an activation is. */
function drawMethod(){
  const host=document.getElementById("methodDiagram"); if(!host) return;
  host.innerHTML="";
  const W=880,H=190;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const box=(x,y,w,h,fill,stroke)=>el("rect",{x,y,width:w,height:h,rx:7,
    fill:fill||"#fff",stroke:stroke||P.border,"stroke-width":1.2});
  const txt=(x,y,t,o={})=>{const e=el("text",{x,y,"text-anchor":o.anchor||"middle",
    "font-size":o.size||11.5,fill:o.fill||P.text,"font-weight":o.weight||400});
    e.textContent=t; return e;};
  const arrow=(x1,x2,y)=>{
    svg.appendChild(el("line",{x1,y1:y,x2:x2-7,y2:y,stroke:P.muted,"stroke-width":1.4}));
    svg.appendChild(el("path",{d:`M${x2-7},${y-4} L${x2},${y} L${x2-7},${y+4}`,
      fill:"none",stroke:P.muted,"stroke-width":1.4}));
  };
  const yMid=78;
  // 1. labelled stories
  [0,1,2].forEach(i=>svg.appendChild(box(24+i*6,44+i*7,120,40,"#fff")));
  svg.appendChild(txt(96,68,"stories tagged",{size:11}));
  svg.appendChild(txt(96,81,"\u201cafraid\u201d",{size:11,weight:600}));
  svg.appendChild(txt(96,124,"1. corpus",{size:10.5,fill:P.muted,weight:600}));
  svg.appendChild(txt(96,138,"~9 or 256 per emotion",{size:9.5,fill:P.muted}));
  arrow(168,214,yMid);
  // 2. forward pass, residual stream tapped at one layer
  svg.appendChild(box(214,38,132,64,"#FAFAFA"));
  [0,1,2,3,4].forEach(i=>{
    const x=228+i*26;
    svg.appendChild(el("rect",{x,y:46,width:15,height:48,rx:3,
      fill:i===3?P.navy:P.border,opacity:i===3?1:.85}));
  });
  svg.appendChild(txt(280,116,"2. forward pass",{size:10.5,fill:P.muted,weight:600}));
  svg.appendChild(txt(280,130,"residual stream at one layer",{size:9.5,fill:P.muted}));
  arrow(352,404,yMid);
  // 3. mean over the corpus, minus the mean over all emotions
  svg.appendChild(box(404,44,150,52,"#fff"));
  svg.appendChild(txt(479,68,"mean over stories",{size:11}));
  svg.appendChild(txt(479,84,"\u2212 mean over emotions",{size:11,fill:P.orange,weight:600}));
  svg.appendChild(txt(479,116,"3. difference of means",{size:10.5,fill:P.muted,weight:600}));
  svg.appendChild(txt(479,130,"centring is load-bearing",{size:9.5,fill:P.orange}));
  arrow(560,596,yMid);
  // 4. the vector, and what it is used for. Shifted left of where this used to
  // sit: the two branch labels on the right ran past the viewBox and were cut.
  svg.appendChild(box(596,44,100,52,"#fff",P.navy));
  svg.appendChild(el("line",{x1:610,y1:82,x2:680,y2:56,stroke:P.navy,"stroke-width":2.2}));
  svg.appendChild(el("path",{d:"M674,54 L681,55 L677,62",fill:"none",stroke:P.navy,"stroke-width":2.2}));
  svg.appendChild(txt(646,116,"4. emotion vector",{size:10.5,fill:P.muted,weight:600}));
  svg.appendChild(txt(646,130,"one direction per emotion",{size:9.5,fill:P.muted}));
  // The two reads that follow. Labels stay short so they fit inside W=880; the
  // paragraph above the figure carries the full definition of the cosine.
  const rx=712;
  svg.appendChild(el("line",{x1:696,y1:yMid,x2:rx-4,y2:yMid,stroke:P.muted,"stroke-width":1.4}));
  svg.appendChild(el("line",{x1:rx-4,y1:44,x2:rx-4,y2:112,stroke:P.muted,"stroke-width":1.4}));
  [[44,"PCA over the 171","\u2192 parts one, two"],
   [112,"cos(stream, vector)","\u2192 per token, part three"]].forEach(([y,a,b])=>{
    arrow(rx-4,rx+12,y);
    svg.appendChild(txt(rx+18,y-2,a,{anchor:"start",size:11,weight:600}));
    svg.appendChild(txt(rx+18,y+12,b,{anchor:"start",size:9.5,fill:P.muted}));
  });
  host.appendChild(svg);
}
drawMethod();

/* ---------- circumplex: principal-component bars ---------- */
/* hostId/verdictId are parameters so the cover can render this exact figure
   rather than a smaller lookalike. A cover chart drawn by separate code is a
   chart that can disagree with the analysis; this one cannot, because it is
   the same function reading the same data. */

/* --- section 5: do the vectors do anything? ---------------------------
   Two reads of the same question, drawn the way the rest of the page draws
   things: every anchor a reader needs to judge the result is in the figure,
   not in the caption. */

/* Read one, correlational. How well a probe's activation predicts which
   activity the model prefers, at each layer it was measured. The registered
   bar and the paper's own range are both drawn, because "0.64" means nothing
   without knowing what was promised and what the paper got. */
function drawPrefLayers(){
  const P0=D.prefs, rows=P0.by_layer, W=780,H=250,L=52,R=210,T=16,B=46;
  const host=document.getElementById("prefLayerChart"); if(!host) return;
  host.innerHTML="";
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B, bw=iw/rows.length;
  // the paper's range, as a band rather than a line: it is a range
  const yOf=v=>T+ih-v*ih;
  svg.appendChild(el("rect",{x:L,y:yOf(P0.paper_range[1]),width:iw,
    height:yOf(P0.paper_range[0])-yOf(P0.paper_range[1]),fill:P.green,opacity:.10}));
  [0,.25,.5,.75,1].forEach(v=>{
    svg.appendChild(el("line",{x1:L,x2:W-R,y1:yOf(v),y2:yOf(v),stroke:P.border}));
    const t=el("text",{x:L-8,y:yOf(v)+4,"text-anchor":"end","font-size":10,fill:P.muted});
    t.textContent=v.toFixed(2); svg.appendChild(t);
  });
  [[P0.registered_bar,"the mark we fixed before scoring",P.orange],
   [P0.paper_range[0],"what the paper reported",P.green]].forEach(([v,label,col])=>{
    svg.appendChild(el("line",{x1:L,x2:W-R,y1:yOf(v),y2:yOf(v),stroke:col,
      "stroke-dasharray":"4 3","stroke-width":1.3}));
    const t=el("text",{x:W-R+8,y:yOf(v)+3.5,"font-size":10.5,fill:col,"font-weight":600});
    t.textContent=v.toFixed(2); svg.appendChild(t);
    const t2=el("text",{x:W-R+38,y:yOf(v)+3.5,"font-size":10,fill:col});
    t2.textContent=label; svg.appendChild(t2);
  });
  rows.forEach((r,i)=>{
    const x0=L+i*bw, h=r.max_abs_r*ih, best=r.layer===P0.best_layer;
    const bar=el("rect",{x:x0+bw*.28,y:yOf(r.max_abs_r),width:bw*.44,height:h,rx:2,
      fill:best?C.valence:P.greyMid});
    tipOn(bar,`<b>layer ${r.layer}</b>: best probe predicts preference at ${r.max_abs_r.toFixed(3)}`+
      `<span class="t-sub">valence organization ${r.valence_r.toFixed(2)}, `+
      `permutation p ${r.perm_p===0?"< 1e-4":r.perm_p}</span>`);
    svg.appendChild(bar);
    const lab=el("text",{x:x0+bw/2,y:H-26,"text-anchor":"middle","font-size":11,fill:P.text});
    lab.textContent="layer "+r.layer; svg.appendChild(lab);
  });
  const yl=el("text",{x:14,y:T+ih/2,"font-size":10.5,fill:P.muted,
    transform:`rotate(-90 14 ${T+ih/2})`,"text-anchor":"middle"});
  yl.textContent="|r| between probe and preference"; svg.appendChild(yl);
  host.appendChild(svg);
}

/* Read two, causal. Add the emotion's own direction to the residual stream and
   see which way preferences move. Emotions run down the axis by their human
   valence rating, so the pattern the claim rests on, negatives pushing down and
   positives pushing up, is the shape of the chart rather than a claim about it.
   Both doses are drawn: one dose could be a fluke, two that scale is a dose
   response. */
function drawSteering(){
  const P0=D.prefs, rows=P0.doses["2"].emotions, W=780,H=380,L=118,R=150,T=18,B=44;
  const host=document.getElementById("steerChart"); if(!host) return;
  host.innerHTML="";
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B, rh=ih/rows.length;
  const byName={};
  P0.doses["8"].emotions.forEach(e=>byName[e.emotion]=e);
  const lim=Math.max(...P0.doses["8"].emotions.map(e=>Math.abs(e.delta)))*1.08;
  const xOf=v=>L+iw/2+(v/lim)*(iw/2);
  [-lim,-lim/2,0,lim/2,lim].forEach(v=>{
    svg.appendChild(el("line",{x1:xOf(v),x2:xOf(v),y1:T,y2:T+ih,
      stroke:v===0?P.text:P.border,"stroke-width":v===0?1.2:1}));
    const t=el("text",{x:xOf(v),y:H-26,"text-anchor":"middle","font-size":10,fill:P.muted});
    t.textContent=Math.round(v); svg.appendChild(t);
  });
  rows.forEach((r,i)=>{
    const y=T+i*rh, hi=byName[r.emotion];
    const agrees=(r.valence>0)===(r.delta>0);
    [[hi.delta,P.greySoft,"strong dose"],[r.delta,r.valence>0?C.dominance:P.red,"gentle dose"]]
      .forEach(([v,fill],k)=>{
        const x=Math.min(xOf(0),xOf(v)), w=Math.abs(xOf(v)-xOf(0));
        const bar=el("rect",{x,y:y+rh*(k?0.42:0.14),width:w,height:rh*(k?0.34:0.28),rx:2,fill});
        tipOn(bar,`<b>${r.emotion}, ${k?"gentle":"strong"} dose</b>: preferences move `+
          `${v>0?"+":""}${v.toFixed(0)} Elo`+
          `<span class="t-sub">human valence ${r.valence>0?"+":""}${r.valence.toFixed(2)}, so this `+
          `${agrees?"moves the way that valence predicts":"moves against its valence"}</span>`);
        svg.appendChild(bar);
      });
    const nm=el("text",{x:L-10,y:y+rh/2+4,"text-anchor":"end","font-size":11.5,
      fill:agrees?P.text:P.alert});
    nm.textContent=r.emotion+(agrees?"":" *"); svg.appendChild(nm);
    const val=el("text",{x:W-R+10,y:y+rh/2+4,"font-size":10,fill:P.muted});
    val.textContent=(r.valence>0?"+":"")+r.valence.toFixed(2); svg.appendChild(val);
  });
  const hdr=el("text",{x:W-R+10,y:T-4,"font-size":9.5,fill:P.muted});
  hdr.textContent="human valence"; svg.appendChild(hdr);
  const xl=el("text",{x:L+iw/2,y:H-8,"text-anchor":"middle","font-size":10.5,fill:P.muted});
  xl.textContent="shift in preference for positive activities, Elo points";
  svg.appendChild(xl);
  host.appendChild(svg);
}

function drawPCs(model, hostId="pcChart", verdictId="pcVerdict"){
  const host=document.getElementById(hostId);
  if(!host) return;                        // no silent half-draw
  host.innerHTML="";
  // R is wide on purpose: the three scale labels live in the right margin,
  // outside the plot. Inside it, the 1.0 label crossed the tallest bars and the
  // 0 label sat on top of the PC5 category tick, whichever side of the line it
  // was placed. A grading anchor that overlaps the data is not an anchor.
  const rows=D.pcs[model], W=780,H=290,L=52,R=168,T=16,B=44;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B, bw=iw/rows.length;
  // grading band: |r| above 0.5 is where a component is meaningfully aligned
  svg.appendChild(el("rect",{x:L,y:T,width:iw,height:ih*0.5,fill:P.green,opacity:.05}));
  [0,.25,.5,.75,1].forEach(v=>{
    const y=T+ih-v*ih;
    svg.appendChild(el("line",{x1:L,x2:W-R,y1:y,y2:y,stroke:P.border}));
    const tx=el("text",{x:L-8,y:y+4,"text-anchor":"end","font-size":10,fill:P.muted});
    tx.textContent=v.toFixed(2); svg.appendChild(tx);
  });
  // both ends of the scale, said in words rather than left to the reader
  [[1,"1.0","exactly that human rating",P.green],
   [0.5,"0.5","clearly related",P.muted],
   [0,"0","nothing to do with it",P.alert]].forEach(([v,num,label,col])=>{
    const y=T+ih-v*ih;
    svg.appendChild(el("line",{x1:L,x2:W-R,y1:y,y2:y,stroke:col,"stroke-dasharray":"4 3",
      "stroke-width":v===0.5?1:1.4,opacity:v===0.5?.5:.9}));
    const t=el("text",{x:W-R+8,y:y+3.5,"font-size":10.5,fill:col,"font-weight":600});
    t.textContent=num; svg.appendChild(t);
    const t2=el("text",{x:W-R+30,y:y+3.5,"font-size":10,fill:col});
    t2.textContent=label; svg.appendChild(t2);
  });
  rows.forEach((r,i)=>{
    const x0=L+i*bw;
    // variance explained as a soft backdrop bar
    const eh=r.evr*ih;
    svg.appendChild(el("rect",{x:x0+8,y:T+ih-eh,width:bw-16,height:eh,fill:C.evr,rx:3}));
    const keys=["valence","arousal","dominance","length"], sw=(bw-26)/keys.length;
    keys.forEach((k,j)=>{
      const h=r[k]*ih;
      const bar=el("rect",{x:x0+13+j*sw,y:T+ih-h,width:sw-3,height:h,fill:C[k],rx:2});
      const nm={valence:"valence",arousal:"arousal",dominance:"dominance",
        length:"story length"}[k];
      tipOn(bar,`<b>axis PC${r.pc} vs human ${nm} ratings</b>: ${r[k].toFixed(2)}`+
        `<span class="t-sub">PC${r.pc} accounts for ${(r.evr*100).toFixed(1)}% of everything `+
        `that separates the 171 emotions.</span>`);
      svg.appendChild(bar);
    });
    const back=el("rect",{x:x0+8,y:T+ih-r.evr*ih,width:bw-16,height:r.evr*ih,fill:"transparent"});
    tipOn(back,`<b>axis PC${r.pc}</b> accounts for ${(r.evr*100).toFixed(1)}% of everything that `+
      `separates the 171 emotions`);
    svg.appendChild(back);
    const lab=el("text",{x:x0+bw/2,y:H-24,"text-anchor":"middle","font-size":11,fill:P.text});
    lab.textContent=r.pc===1?"PC1 (biggest axis)":"PC"+r.pc; svg.appendChild(lab);
    const ev=el("text",{x:x0+bw/2,y:H-10,"text-anchor":"middle","font-size":9.5,fill:P.muted});
    ev.textContent=(r.evr*100).toFixed(1)+"% of the spread"; svg.appendChild(ev);
  });
  // the y axis had tick numbers but never said what they measured
  const yl=el("text",{x:14,y:T+ih/2,"font-size":10.5,fill:P.muted,
    transform:`rotate(-90 14 ${T+ih/2})`,"text-anchor":"middle"});
  yl.textContent="|r| with the human rating"; svg.appendChild(yl);
  host.appendChild(svg);
  const verdict=document.getElementById(verdictId);
  if(verdict) verdict.textContent = model==="base"
    ? "biggest axis = valence, 0.83. the circumplex, recovered."
    : "biggest axis = unknown. valence scores 0.11 here, and leads PC3 instead.";
}
document.querySelectorAll("[data-model]").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("[data-model]").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); drawPCs(b.dataset.model);
});

/* ---------- what moved in: correlation grid ---------- */
function drawGrid(){
  const host=document.getElementById("gridChart"); host.innerHTML="";
  const W=420,H=420,L=64,T=42,cell=62;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const note=document.getElementById("gridNote");
  D.grid.forEach((row,i)=>row.forEach((v,j)=>{
    const x=L+j*cell,y=T+i*cell;
    const g=el("rect",{x,y,width:cell-3,height:cell-3,rx:3,
      fill:`rgba(29,53,87,${(v*0.95).toFixed(3)})`,style:"cursor:pointer"});
    tipOn(g,`<b>instruction-tuned axis ${i+1} vs base axis ${j+1}</b>: ${v.toFixed(2)}`+
      `<span class="t-sub">`+
      (i===0 ? "This row is the test. The instruction-tuned model's biggest axis scores at most 0.14 against any of the base model's five biggest axes, so it is new structure rather than a rearrangement of them." :
       i===2&&j===0 ? "This is valence: the base model's top axis, still intact, but demoted to third place by instruction tuning." :
       "1 would mean the two axes carry the same information; 0 that they have nothing in common.")+
      `</span>`);
    svg.appendChild(g);
    const t=el("text",{x:x+(cell-3)/2,y:y+(cell-3)/2+4,"text-anchor":"middle","font-size":11,
      fill:v>0.45?"#fff":P.text}); t.textContent=v.toFixed(2); svg.appendChild(t);
  }));
  // column and row headers, plus one line naming which model each side belongs to,
  // because "it PC1" read as an abbreviation nobody on the page had defined
  const ch=el("text",{x:L,y:14,"font-size":10.5,fill:P.muted,"font-weight":600});
  ch.textContent="the base model's five biggest axes"; svg.appendChild(ch);
  for(let j=0;j<5;j++){const t=el("text",{x:L+j*cell+(cell-3)/2,y:T-12,"text-anchor":"middle",
    "font-size":10.5,fill:P.muted});t.textContent="axis "+(j+1);svg.appendChild(t);}
  for(let i=0;i<5;i++){const t=el("text",{x:L-8,y:T+i*cell+(cell-3)/2+4,"text-anchor":"end",
    "font-size":10.5,fill:P.muted});t.textContent="axis "+(i+1);svg.appendChild(t);}
  const rh=el("text",{x:12,y:T+(5*cell)/2,"font-size":10.5,fill:P.muted,"font-weight":600,
    transform:`rotate(-90 12 ${T+(5*cell)/2})`,"text-anchor":"middle"});
  rh.textContent="the instruction-tuned model's five biggest axes"; svg.appendChild(rh);
  // a scale strip, so a shade can be read without hovering
  const sx=L, sy=T+5*cell+14, sw=5*cell-3;
  for(let i=0;i<40;i++){
    svg.appendChild(el("rect",{x:sx+i*(sw/40),y:sy,width:sw/40+.5,height:9,
      fill:`rgba(29,53,87,${((i/39)*0.95).toFixed(3)})`}));
  }
  // only the two ends are labelled: a middle tick collided with the right label
  [["0 = nothing in common",0,"start"],["1 = the same information",1,"end"]].forEach(([lab,f,anc])=>{
    const t=el("text",{x:sx+f*sw,y:sy+22,"text-anchor":anc,"font-size":10,fill:P.muted});
    t.textContent=lab; svg.appendChild(t);
  });
  host.appendChild(svg);
  note.textContent="hover any square for what that pairing means";
}
document.getElementById("pc1low").textContent=D.itpc1.low.join(", ");
document.getElementById("pc1high").textContent=D.itpc1.high.join(", ");

/* ---------- one story, token by token ---------- */
const EC=[P.red,P.teal,P.green];
/* Every story this section can show: the walkthrough, plus three stories that
   share the SAME three emotions and differ only in how they are written. Those
   three were picked by measured tracking quality (best, median and worst mean
   gate rank at layer 33), not by eye. */
const STORIES=[
  {key:"orig", label:"the walkthrough", qual:null, id:D.story.story_id,
   emotions:D.story.emotions, boundaries:D.story.boundaries, n_tokens:D.story.n_tokens,
   lines_by_layer:D.story.lines_by_layer, text:D.storyText.text},
  ...D.three.map(t=>({
    key:t.quality,
    label:t.quality==="strong"?"tracked well":(t.quality==="mixed"?"tracked partly":"tracked badly"),
    qual:{wins:t.wins, perPhase:t.per_phase_win, margin:t.mean_margin}, id:t.id,
    emotions:t.emotions, boundaries:t.boundaries, n_tokens:t.n_tokens,
    lines_by_layer:t.lines_by_layer, text:t.text}))
];
let curStory=0, S=STORIES[0];
let curLayer=D.story.default_layer, curTok=0;
function drawStory(){
  const ys=S.lines_by_layer[curLayer];
  // line chart
  const host=document.getElementById("lineChart"); host.innerHTML="";
  const W=470,H=300,L=62,R=12,T=14,B=44;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B, n=S.n_tokens;
  let lo=Infinity,hi=-Infinity; ys.forEach(s=>s.forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v);}));
  const pad=(hi-lo)*0.08; lo-=pad; hi+=pad;
  const X=i=>L+(i/(n-1))*iw, Y=v=>T+ih-((v-lo)/(hi-lo))*ih;
  // the zero line is the story's own average: above it the model leans toward
  // that emotion, below it away. Labelled, because an unlabelled zero is noise.
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(0),y2:Y(0),stroke:P.greyMid,"stroke-dasharray":"3 3"}));
  // in the left margin, not over the plot: the three lines cross the zero line
  // constantly, so an inline label sat on top of the data for most stories
  [[hi,hi.toFixed(2)],[0,"0"],[lo,lo.toFixed(2)]].forEach(([v,lab])=>{
    const t=el("text",{x:L-6,y:Y(v)+3,"text-anchor":"end","font-size":9,
      fill:v===0?P.text:P.muted});
    t.textContent=lab; svg.appendChild(t);
  });
  const zl2=el("text",{x:L-6,y:Y(0)+12,"text-anchor":"end","font-size":8,fill:P.muted});
  zl2.textContent="corpus avg"; svg.appendChild(zl2);
  S.boundaries.forEach(b=>{
    svg.appendChild(el("line",{x1:X(b),x2:X(b),y1:T,y2:T+ih,stroke:P.text,"stroke-width":1,"stroke-dasharray":"4 3",opacity:.45}));
  });
  ys.forEach((serie,k)=>{
    let d="";serie.forEach((v,i)=>{d+=(i?"L":"M")+X(i).toFixed(1)+","+Y(v).toFixed(1);});
    svg.appendChild(el("path",{d,fill:"none",stroke:EC[k],"stroke-width":1.9,opacity:.9}));
    const hit=el("path",{d,fill:"none",stroke:"transparent","stroke-width":11});
    tipOn(hit,()=>`<b>${S.emotions[k]}</b> at token ${curTok}: ${serie[curTok].toFixed(3)}`+
      `<span class="t-sub">how close the model's state sits to the ${S.emotions[k]} vector. `+
      `Higher means closer. Read at layer ${curLayer}.</span>`);
    svg.appendChild(hit);
  });
  svg.appendChild(el("line",{x1:X(curTok),x2:X(curTok),y1:T,y2:T+ih,stroke:P.orange,"stroke-width":2}));
  ys.forEach((serie,k)=>svg.appendChild(el("circle",{cx:X(curTok),cy:Y(serie[curTok]),r:4.5,
    fill:EC[k],stroke:"#fff","stroke-width":1.5})));
  const readout=document.getElementById("storyReadout");
  if(readout) readout.innerHTML = S.emotions.map((nm,k)=>
    `<span><i style="background:${EC[k]}"></i>${nm} <b style="color:${EC[k]}">`+
    `${ys[k][curTok].toFixed(3)}</b></span>`).join("")+
    `<span style="margin-left:auto">reading at token ${curTok}, layer ${curLayer}</span>`;
  [[0,"0"],[S.boundaries[0],"turn 1 · "+S.boundaries[0]],
   [S.boundaries[1],"turn 2 · "+S.boundaries[1]],[n-1,String(n-1)]].forEach(([i,lab],k)=>{
    const t=el("text",{x:X(i),y:T+ih+13,"text-anchor":k===0?"start":(k===3?"end":"middle"),
      "font-size":9,fill:k===1||k===2?P.text:P.muted});
    t.textContent=lab; svg.appendChild(t);
  });
  const xl=el("text",{x:L+iw/2,y:H-6,"text-anchor":"middle","font-size":10.5,fill:P.muted});
  xl.textContent="token in the story";
  svg.appendChild(xl);
  const yl=el("text",{x:12,y:T+ih/2,"font-size":10.5,fill:P.muted,
    transform:`rotate(-90 12 ${T+ih/2})`,"text-anchor":"middle"});
  yl.textContent="closeness to each emotion vector"; svg.appendChild(yl);
  // the y range is re-fitted per layer, so say so rather than let the reader
  // assume the shapes are comparable across the layer buttons
  const rn=el("text",{x:W-R,y:T+8,"text-anchor":"end","font-size":8.5,fill:P.muted});
  rn.textContent="y range re-fitted per layer"; svg.appendChild(rn);
  host.appendChild(svg);

  // ternary
  const th=document.getElementById("ternChart"); th.innerHTML="";
  const TW=470,TH=316,cx=TW/2,top=26,side=232,hgt=side*Math.sin(Math.PI/3);
  const s2=el("svg",{viewBox:`0 0 ${TW} ${TH}`,width:"100%"});
  const A=[cx,top], B2=[cx-side/2,top+hgt], Cc=[cx+side/2,top+hgt];
  s2.appendChild(el("polygon",{points:`${A[0]},${A[1]} ${B2[0]},${B2[1]} ${Cc[0]},${Cc[1]}`,
    fill:"#FAFAFA",stroke:P.border}));
  const proj=(a,b,c)=>{const s=a+b+c||1;a/=s;b/=s;c/=s;
    return [a*A[0]+b*B2[0]+c*Cc[0], a*A[1]+b*B2[1]+c*Cc[1]];};
  // the triangle position comes from the same three curves, shifted to be
  // non-negative then normalised, so every story can be shown the same way
  const floor=Math.min(...ys.flat()), shift=v=>v-floor+1e-3;
  const mix=i=>[shift(ys[0][i]),shift(ys[1][i]),shift(ys[2][i])];
  // Only the path already walked is drawn. Drawing the whole story from t=0
  // showed the reader the future and made the walk look directionless.
  let dPast="", dRest="";
  for(let i=0;i<S.n_tokens;i++){const m=mix(i);const p=proj(m[0],m[1],m[2]);
    const seg=p[0].toFixed(1)+","+p[1].toFixed(1);
    if(i<=curTok) dPast+=(i?"L":"M")+seg;
    if(i>=curTok) dRest+=(i===curTok?"M":"L")+seg;}
  // what is still to come, barely visible, so the shape is not a surprise
  s2.appendChild(el("path",{d:dRest,fill:"none",stroke:P.border,"stroke-width":1.2,
    "stroke-dasharray":"2 3",opacity:.6}));
  s2.appendChild(el("path",{d:dPast,fill:"none",stroke:P.muted,"stroke-width":1.6,opacity:.75}));
  for(let i=0;i<=curTok;i+=3){const m=mix(i);const p=proj(m[0],m[1],m[2]);
    const ph=i<S.boundaries[0]?0:(i<S.boundaries[1]?1:2);
    s2.appendChild(el("circle",{cx:p[0],cy:p[1],r:2.4,fill:EC[ph],opacity:.5}));}
  const mc=mix(curTok); const pc=proj(mc[0],mc[1],mc[2]);
  const phase=curTok<S.boundaries[0]?0:(curTok<S.boundaries[1]?1:2);
  const dot=el("circle",{cx:pc[0],cy:pc[1],r:8,fill:EC[phase],stroke:"#fff","stroke-width":2.5});
  tipOn(dot,()=>{
    const m=mix(curTok), tot=m[0]+m[1]+m[2];
    const parts=S.emotions.map((nm,k)=>`${nm} ${pct(m[k]/tot)}`);
    return `<b>token ${curTok}, written as ${S.emotions[phase]}</b>`+
      `<span class="t-sub">the model's state reads as: ${parts.join(", ")}.`+
      `<br>A corner means it looks purely like that emotion; the middle means undecided.</span>`;
  });
  s2.appendChild(dot);
  const corners=[[A,S.emotions[0],"middle",-10],[B2,S.emotions[1],"end",16],[Cc,S.emotions[2],"start",16]];
  corners.forEach(([p,nm,anc,dy],k)=>{
    const t=el("text",{x:p[0]+(anc==="end"?-6:anc==="start"?6:0),y:p[1]+dy,"text-anchor":anc,
      "font-size":12,fill:EC[k],"font-weight":600}); t.textContent=nm; s2.appendChild(t);
  });
  // what the middle of the triangle means, said on the figure
  const mid=proj(1,1,1);
  s2.appendChild(el("circle",{cx:mid[0],cy:mid[1],r:3,fill:"none",stroke:P.muted,
    "stroke-dasharray":"2 2"}));
  const cl2=el("text",{x:cx,y:TH-38,"text-anchor":"middle","font-size":9.5,fill:P.muted});
cl2.textContent="corner = reads purely as that emotion · middle = undecided"; s2.appendChild(cl2);
  const cap=el("text",{x:cx,y:TH-23,"text-anchor":"middle","font-size":9.5,fill:P.muted});
cap.textContent="solid line = the walk so far · dotted = still to come · "+
    "big dot = where it is now"; s2.appendChild(cap);
  const cap2=el("text",{x:cx,y:TH-7,"text-anchor":"middle","font-size":10.5,fill:P.muted});
  cap2.textContent="written to walk "+S.emotions.join(" → "); s2.appendChild(cap2);
  th.appendChild(s2);

  document.getElementById("tokLabel").textContent="t = "+curTok+" / "+(S.n_tokens-1);
  paintStory(phase);
}

/* the story itself, split on its own <emotion> markers, dimmed except the live phase */
function storyPhases(raw){
  const out=[];
  const re=/<emotion>([^<]+)<\/emotion>/g; let m, last=null, lastIdx=0;
  while((m=re.exec(raw))!==null){
    if(last!==null) out.push({label:last,text:raw.slice(lastIdx,m.index).trim()});
    last=m[1]; lastIdx=re.lastIndex;
  }
  if(last!==null) out.push({label:last,text:raw.slice(lastIdx).trim()});
  return out;
}
function paintStory(active){
  const box=document.getElementById("storyBox");
  box.innerHTML=storyPhases(S.text).map((p,i)=>
    `<div class="ph${i===active?"":" dim"}">`+
    `<span class="phtag" style="color:${EC[i]}">phase ${i+1} &middot; written as ${p.label}</span>`+
    p.text.replace(/\n+/g,"<br>")+`</div>`).join("");
}
function selectStory(i){
  curStory=i; S=STORIES[i];
  curTok=0;
  const sl=document.getElementById("tokSlider");
  sl.max=S.n_tokens-1; sl.value=0;
  const q=S.qual
    ? `${S.qual.wins} of 3 phases led by the right emotion (per phase: `+
      `${S.qual.perPhase.map(w=>w?"yes":"no").join(", ")}); average margin `+
      `${S.qual.margin>=0?"+":""}${S.qual.margin.toFixed(3)}. One of three by luck.`
    : "the worked example; the other three are scored, best / middle / worst of a random 24";
  document.getElementById("storyQual").textContent=q;
  document.getElementById("storyId").textContent=S.id;
  drawStory();
}
(function(){
  const sb=document.getElementById("storyBtns");
  STORIES.forEach((st,i)=>{
    const b=document.createElement("button");
    b.className="seg"+(i===0?" on":""); b.textContent=st.label;
    b.onclick=()=>{sb.querySelectorAll("button").forEach(x=>x.classList.remove("on"));
      b.classList.add("on"); selectStory(i);};
    sb.appendChild(b);
  });
})();
/* Play control for the word slider. Reading a 236-word story by dragging it is
   work, and the thing worth seeing, the handover from one emotion to the next,
   is a motion rather than a position. It loops instead of stopping at the end,
   because you almost always want the handover twice. The slider stays: playback
   is for watching, the slider is for going back to the word that surprised you.

   Declared at top level, above the block that wires the button. An earlier
   version sat inside that block, which put `storyTimer` in a scope the handlers
   could not see and silently blanked every figure on the page. */
const PLAY_MS = 80;
const PLAY_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3l8 5-8 5z"/></svg>';
const PAUSE_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5.5 3v10M10.5 3v10"/></svg>';
let storyTimer = null;
function paintPlayButton(){
  const b=document.getElementById("tokPlay"); if(!b) return;
  b.innerHTML = storyTimer ? PAUSE_ICON : PLAY_ICON;
  b.setAttribute("aria-label", storyTimer ? "Pause" : "Play the story");
  b.classList.toggle("on", !!storyTimer);
}
function stopStory(){
  if(storyTimer){ clearInterval(storyTimer); storyTimer=null; }
  paintPlayButton();
}
function playStory(){
  const sl=document.getElementById("tokSlider"); if(!sl || storyTimer) return;
  storyTimer=setInterval(()=>{
    curTok = curTok >= +sl.max ? 0 : curTok+1;
    sl.value=curTok; drawStory();
  }, PLAY_MS);
  paintPlayButton();
}

(function(){
  const lb=document.getElementById("layerBtns");
  Object.keys(STORIES[0].lines_by_layer).forEach(L=>{
    const b=document.createElement("button"); b.className="seg"+(L===curLayer?" on":"");
    b.textContent=L; b.onclick=()=>{curLayer=L;
      lb.querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); drawStory();};
    lb.appendChild(b);
  });
  const sl=document.getElementById("tokSlider");
  sl.max=S.n_tokens-1;
  // taking the slider means taking control: playback stops rather than fighting the drag
  sl.oninput=()=>{stopStory(); curTok=+sl.value; drawStory();};
  const pb=document.getElementById("tokPlay");
  if(pb){ pb.onclick=()=>storyTimer?stopStory():playStory(); tipOn(pb,"Play the story from the first token"); pb.style.cursor="pointer"; }
  paintPlayButton();
})();

/* ---------- per emotion, every layer ---------- */
let emoBank="selfgen", emoLayer="33";
function drawEmo(){
  const host=document.getElementById("emoChart"); host.innerHTML="";
  const rows=D.emoByLayer[emoBank][emoLayer];
  const W=880,H=330,L=64,R=150,T=20,B=70;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B, bw=iw/rows.length;
  // Fixed 0-100% domain. It used to rescale to the tallest bar, so switching
  // vector set redrew 49% and 83% at almost the same length — on the one
  // control the section exists to make you compare.
  const chance=1/12, Y=v=>T+ih-v*ih;
  [0,0.25,0.5,0.75,1].forEach(v=>{
    svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:P.border}));
    const t=el("text",{x:L-7,y:Y(v)+3.5,"text-anchor":"end","font-size":9.5,fill:P.muted});
    t.textContent=(v*100)+"%"; svg.appendChild(t);
  });
  const yl=el("text",{x:16,y:T+ih/2,"font-size":10.5,fill:P.muted,
    transform:`rotate(-90 16 ${T+ih/2})`,"text-anchor":"middle"});
  yl.textContent="phases where this emotion’s own vector wins"; svg.appendChild(yl);
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(chance),y2:Y(chance),stroke:P.alert,
    "stroke-dasharray":"5 4","stroke-width":1.5}));
  // in the right margin, clear of the bars: inside the plot this label was
  // drawn in dark red across the two tallest navy bars
  const ct=el("text",{x:W-R+6,y:Y(chance)+3.5,"font-size":10.5,fill:P.alert});
  ct.textContent="chance (8%)"; svg.appendChild(ct);
  const ct2=el("text",{x:W-R+6,y:Y(chance)+16,"font-size":9.5,fill:P.muted});
  ct2.textContent="at or below this line: no signal"; svg.appendChild(ct2);
  const gt=el("text",{x:W-R+6,y:Y(1)+3.5,"font-size":10.5,fill:P.green});
  gt.textContent="100% = always right"; svg.appendChild(gt);
  rows.forEach((r,i)=>{
    const x=L+i*bw, h=Math.max(0,(T+ih)-Y(r.rate));
    const bar=el("rect",{x:x+7,y:r.rate>0?Y(r.rate):T+ih-3,width:bw-14,height:r.rate>0?h:3,
      rx:r.rate>0?3:1,fill:r.rate>=chance?P.navy:(r.rate>0?P.greyMid:P.alert)});
    const wrongHtml=r.wrong.length
      ? r.wrong.map(w=>`${w[0]} ${pct(w[1])}`).join(", ")
      : "no single dominant wrong answer";
    tipOn(bar,`<b>${r.e}</b> at layer ${emoLayer}`+
      `<span class="t-sub">its own probe wins <b>${pct(r.rate)}</b> of ${r.n} story phases`+
      ` (chance is 1/12).<br>When it is wrong, the model says: ${wrongHtml}.</span>`);
    svg.appendChild(bar);
    const nearChance=Math.abs(r.rate-chance)<0.03;
    const v=el("text",{x:x+bw/2,y:Y(r.rate)+(nearChance?-14:-6),"text-anchor":"middle",
      "font-size":10.5,fill:r.rate>0?P.text:P.alert});
    v.textContent=r.rate>0?pct(r.rate):"never"; svg.appendChild(v);
    const t=el("text",{x:x+bw/2,y:T+ih+16,"text-anchor":"end","font-size":11,fill:P.body,
      transform:`rotate(-40 ${x+bw/2} ${T+ih+16})`}); t.textContent=r.e; svg.appendChild(t);
  });
  host.appendChild(svg);
  // the standing note restates the verdict for whichever layer and vector set is showing
  const best=rows[0], nWin=rows.filter(r=>r.rate>=chance).length, nNever=rows.filter(r=>r.rate===0).length;
  document.getElementById("emoNote").innerHTML =
    `<b>layer ${emoLayer}, vectors from ${emoBank==="selfgen"?"Gemma's own":"DeepSeek's"} stories:</b> `+
`${nWin} of 12 emotions beat chance, best is ${best.e} at ${pct(best.rate)}`+
    ` (${rows[0].n} phases per emotion)`+
    (nNever?`, and ${nNever} never win at all`:"")+". Hover any bar for its wrong answers.";
}
(function(){
  const host=document.getElementById("emoLayerBtns");
  Object.keys(D.emoByLayer.selfgen).forEach(L=>{
    const b=document.createElement("button");
    b.className="seg"+(L===emoLayer?" on":""); b.textContent=L;
    b.onclick=()=>{emoLayer=L;
      host.querySelectorAll("button").forEach(x=>x.classList.remove("on"));
      b.classList.add("on"); drawEmo();};
    host.appendChild(b);
  });
})();
document.querySelectorAll("[data-vectors]").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("[data-vectors]").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); emoBank=b.dataset.vectors; drawEmo();
});

/* ---------- naming vs anticipating, by layer ---------- */
function drawLayers(){
  const host=document.getElementById("layerChart"); host.innerHTML="";
  const rows=D.byLayer, W=470,H=320,L=52,R=52,T=34,B=44;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B;
  const X=i=>L+(i/(rows.length-1))*iw;
  const Y1=v=>T+ih-(v/0.65)*ih, Y2=v=>T+ih-(v/0.30)*ih;
  // Two measurements on two scales share this chart, so each line gets its own
  // axis title in its own colour and its own labelled failure line. Without the
  // colour pairing a reader cannot tell which y-value belongs to which curve.
  const yLeft=el("text",{x:13,y:T+ih/2,"font-size":10,fill:P.navy,
    transform:`rotate(-90 13 ${T+ih/2})`,"text-anchor":"middle"});
  yLeft.textContent="how often it names the emotion right"; svg.appendChild(yLeft);
  const yRight=el("text",{x:W-13,y:T+ih/2,"font-size":10,fill:P.orange,
    transform:`rotate(90 ${W-13} ${T+ih/2})`,"text-anchor":"middle"});
  yRight.textContent="lean size vs size of the coming change"; svg.appendChild(yRight);
  [[0,"0%"],[0.3,"30%"],[0.6,"60%"]].forEach(([v,lab])=>{
    const t=el("text",{x:L-7,y:Y1(v)+3.5,"text-anchor":"end","font-size":9.5,fill:P.navy});
    t.textContent=lab; svg.appendChild(t);
  });
  [[0,"0"],[0.15,"+0.15"],[0.30,"+0.30"]].forEach(([v,lab])=>{
    const t=el("text",{x:W-R+7,y:Y2(v)+3.5,"font-size":9.5,fill:P.orange});
    t.textContent=lab; svg.appendChild(t);
  });
  // anchors: chance for the naming curve, zero for the anticipation curve
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y1(1/12),y2:Y1(1/12),stroke:P.navy,
    "stroke-dasharray":"4 3",opacity:.55}));
  const a1=el("text",{x:W-R-2,y:Y1(1/12)-6,"text-anchor":"end","font-size":9.5,fill:P.navy});
  a1.textContent="chance (8%)"; svg.appendChild(a1);
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y2(0),y2:Y2(0),stroke:P.orange,
    "stroke-dasharray":"4 3",opacity:.55}));
  // above its own line, not below: below puts it on the x-axis tick labels, and
  // both zero lines land at the same height because the two scales share a floor
  const a2=el("text",{x:W-R-2,y:Y2(0)-6,"text-anchor":"end","font-size":9.5,fill:P.orange});
  // kept short on purpose: right-anchored, this label extends leftwards into the
  // plot, and layer 33 sits exactly ON the zero line — a longer string runs
  // straight under that point. The axis title carries the full meaning.
  a2.textContent="0 = no relation"; svg.appendChild(a2);
  let d1="",d2="";
  rows.forEach((r,i)=>{d1+=(i?"L":"M")+X(i)+","+Y1(r.top1);d2+=(i?"L":"M")+X(i)+","+Y2(r.r_dval);});
  svg.appendChild(el("path",{d:d1,fill:"none",stroke:P.navy,"stroke-width":2.4}));
  svg.appendChild(el("path",{d:d2,fill:"none",stroke:P.orange,"stroke-width":2.4,"stroke-dasharray":"5 3"}));
  rows.forEach((r,i)=>{
    const c1=el("circle",{cx:X(i),cy:Y1(r.top1),r:7,fill:P.navy});
    tipOn(c1,`<b>layer ${r.layer}: names the right emotion ${pct(r.top1)} of the time</b>`+
      `<span class="t-sub">when it is wrong, the emotion it picks instead sits ${r.vad.toFixed(2)} `+
      `away in the VAD space, against ${r.shuffle.toFixed(2)} for a `+
      `randomly chosen emotion: wrong, but nearby.</span>`);
    svg.appendChild(c1);
    const c2=el("circle",{cx:X(i),cy:Y2(r.r_dval),r:7,fill:P.orange});
    tipOn(c2,`<b>layer ${r.layer}: anticipation +${r.r_dval.toFixed(3)}</b>`);
    svg.appendChild(c2);
    const t=el("text",{x:X(i),y:H-24,"text-anchor":"middle","font-size":11,fill:P.text});
    t.textContent="L"+r.layer; svg.appendChild(t);
  });
  const a=el("text",{x:X(0),y:Y1(rows[0].top1)-10,"font-size":10.5,fill:P.navy});
  a.textContent="58% at layer 6"; svg.appendChild(a);
  const b=el("text",{x:X(5),y:Y2(rows[5].r_dval)-10,"text-anchor":"end","font-size":10.5,fill:P.orange});
  b.textContent="+0.26 at layer 51"; svg.appendChild(b);
  const xl=el("text",{x:L+iw/2,y:H-8,"text-anchor":"middle","font-size":10.5,fill:P.muted});
  xl.textContent="layer of the model"; svg.appendChild(xl);
  host.appendChild(svg);
}


/* ---------- probe lineage: who wrote the stories ---------- */
/* Grid, not bars. The old chart showed "passing layers", a count of a count:
   a layer passes when >=8 of 12 chat scenarios put the target emotion in the
   top 3 of 12, on BOTH batteries. Neither level was visible, so the number had
   to be taken on trust. Here every score is drawn and the count is the number
   of ringed cells in a row. */
function drawLineage(){
  const host=document.getElementById("lineageChart"); if(!host) return;
  host.innerHTML="";
  const LL=D.lineageLayers, layers=LL.layers, rows=D.lineage;
  const W=880,H=300,L=232,R=104,T=78,B=48;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B;
  const cw=iw/layers.length, rh=ih/rows.length;
  const shade=v=>`rgba(29,53,87,${(0.06+0.94*Math.min(1,v/12)).toFixed(3)})`;

  const ttl=el("text",{x:14,y:18,"font-size":13,fill:P.text,"font-weight":600});
  ttl.textContent="Can these emotion vectors spot the emotion behind a user's message?";
  svg.appendChild(ttl);
  const sub=el("text",{x:14,y:34,"font-size":10.5,fill:P.muted});
  sub.textContent="cell = of 12 scenarios, how many put the right emotion in its top 3";
  svg.appendChild(sub);

  rows.forEach((r,i)=>{
    const arm=LL.arms[r.key], y=T+i*rh;
    const nm=el("text",{x:L-12,y:y+rh/2-2,"text-anchor":"end","font-size":11.5,fill:P.text});
    nm.textContent=r.label; svg.appendChild(nm);
    const sb=el("text",{x:L-12,y:y+rh/2+11,"text-anchor":"end","font-size":10,fill:P.muted});
    // corpus size on the row, because it is the confound: the four arms span
    // 1,539 to 12,262 stories, so this is not a single-variable comparison
    sb.textContent=`${r.sub} · ${r.n.toLocaleString()} stories`; svg.appendChild(sb);
    layers.forEach((layer,j)=>{
      const paper=arm.paper[j], held=arm.heldout[j], worst=Math.min(paper,held);
      const passes=paper>=LL.bar && held>=LL.bar;
      const x=L+j*cw;
      const cell=el("rect",{x:x+1,y:y+3,width:cw-2,height:rh-6,rx:2,fill:shade(worst)});
      tipOn(cell,`<b>${r.label}, layer ${layer}</b>: `+
        `${paper} of 12 on the paper's scenarios, ${held} of 12 on ours`+
        `<span class="t-sub">${passes?"passes":"fails"} the ${LL.bar}-of-12 mark, `+
        `which has to be met on both.</span>`);
      svg.appendChild(cell);
      // The count in the cell, not only in the hover text. Shade alone cannot
      // separate 7 from 8, and 8 is the pass mark; hover also reaches nobody on
      // a touch screen, on a keyboard, or inside the expanded view, which
      // clones the svg without its listeners.
      const num=el("text",{x:x+cw/2,y:y+rh/2+4,"text-anchor":"middle","font-size":11,
        "font-weight":600,fill:worst/12>0.55?"#fff":P.text});
      num.textContent=worst; svg.appendChild(num);
      if(passes) svg.appendChild(el("rect",{x:x+1,y:y+3,width:cw-2,height:rh-6,rx:2,
        fill:"none",stroke:P.orange,"stroke-width":2}));
    });
    // the bar chart's number, now just the ringed cells in this row
    const cnt=el("text",{x:W-R+10,y:y+rh/2-1,"font-size":15,fill:P.orange,"font-weight":700});
    cnt.textContent=arm.passing.length; svg.appendChild(cnt);
    const cl=el("text",{x:W-R+30,y:y+rh/2-1,"font-size":10.5,fill:P.muted});
    cl.textContent="of 20 layers"; svg.appendChild(cl);
  });

  // layer axis
  layers.forEach((layer,j)=>{
    if(j%3) return;
    const t=el("text",{x:L+j*cw+cw/2,y:T+ih+15,"text-anchor":"middle","font-size":9.5,fill:P.muted});
    t.textContent=layer; svg.appendChild(t);
  });
  const xl=el("text",{x:L+iw/2,y:H-6,"text-anchor":"middle","font-size":10.5,fill:P.muted});
  xl.textContent="layer of the model (20 sampled, every third from 0 to 57)"; svg.appendChild(xl);

  // colour key, with the pass mark on it
  const kx=L, ky=T-26, kw=140;
  for(let i=0;i<30;i++)
    svg.appendChild(el("rect",{x:kx+i*(kw/30),y:ky,width:kw/30+.5,height:8,
      fill:shade((i/29)*12)}));
  const k0=el("text",{x:kx-4,y:ky+7,"text-anchor":"end","font-size":9,fill:P.muted});
  k0.textContent="0"; svg.appendChild(k0);
  const k12=el("text",{x:kx+kw+4,y:ky+7,"font-size":9,fill:P.muted});
  k12.textContent="12 of 12"; svg.appendChild(k12);
  const bx=kx+(LL.bar/12)*kw;
  svg.appendChild(el("line",{x1:bx,x2:bx,y1:ky-3,y2:ky+11,stroke:P.orange,"stroke-width":1.6}));
  const bl=el("text",{x:kx+kw+62,y:ky+7,"font-size":9.5,fill:P.orange,"font-weight":600});
  bl.textContent="↑ 8 = pass mark, fixed before scoring"; svg.appendChild(bl);
  const rl=el("text",{x:W-R+10,y:ky+7,"font-size":9.5,fill:P.orange});
  rl.textContent="ringed = passes"; svg.appendChild(rl);

  host.appendChild(svg);
}

/* dose-response: how many stories per emotion you actually need */
function drawDose(){
  const host=document.getElementById("doseChart"); host.innerHTML="";
  const ns=Object.keys(D.dose), W=380,H=190,L=52,R=14,T=18,B=42;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B;
  const X=i=>L+(i/(ns.length-1))*iw, Y=v=>T+ih-(v/20)*ih;
  // This chart shares its y-scale with the bar chart beside it but used to draw
  // no axis at all, so its dots floated free of the "9 of 20" they refer to.
  [0,10,20].forEach(v=>{
    svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:P.border}));
    const t=el("text",{x:L-6,y:Y(v)+3.5,"text-anchor":"end","font-size":9.5,fill:P.muted});
    t.textContent=v; svg.appendChild(t);
  });
  const yl=el("text",{x:14,y:T+ih/2,"font-size":9.5,fill:P.muted,
    transform:`rotate(-90 14 ${T+ih/2})`,"text-anchor":"middle"});
  yl.textContent="layers that work, of 20"; svg.appendChild(yl);
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(9),y2:Y(9),stroke:P.green,
    "stroke-dasharray":"4 3"}));
  // left-anchored: right-anchored, the trailing "9" sat under the last marker
  const cl=el("text",{x:L+4,y:Y(9)-5,"font-size":9.5,fill:P.green});
  cl.textContent="best this writer ever reaches: 9 of 20"; svg.appendChild(cl);
  let d=""; ns.forEach((n,i)=>{d+=(i?"L":"M")+X(i)+","+Y(D.dose[n]);});
  svg.appendChild(el("path",{d,fill:"none",stroke:P.navy,"stroke-width":2.2}));
  ns.forEach((n,i)=>{
    const c=el("circle",{cx:X(i),cy:Y(D.dose[n]),r:6,fill:P.navy});
    tipOn(c,`<b>${n} stories per emotion</b><span class="t-sub">gives ${D.dose[n]} working layers, `+
      `averaged over 5 random draws of that many stories. The most this writer ever reaches is 9.</span>`);
    svg.appendChild(c);
    const t=el("text",{x:X(i),y:H-16,"text-anchor":"middle","font-size":10,fill:P.muted});
    t.textContent=n; svg.appendChild(t);
  });
  const xl=el("text",{x:L+iw/2,y:H-3,"text-anchor":"middle","font-size":10,fill:P.muted});
  xl.textContent="stories per emotion"; svg.appendChild(xl);
  host.appendChild(svg);
}


/* ---------- tabbed math / pseudo-code blocks (site CodeTabs pattern) ---------- */
const PY_KW=new Set(["def","for","in","return","if","else","import","from","as","not","and","or",
  "None","True","False","while","with","lambda","assert","break","continue"]);
function hlPython(code){
  const re=/(#[^\n]*)|("(?:[^"\\]|\\.)*")|(\b\d+\.?\d*\b)|(\b[A-Za-z_]\w*\b)|(\s+)|([^\s\w])/g;
  let out="",m;
  while((m=re.exec(code))!==null){
    const esc=t=>t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    if(m[1]) out+=`<span class="k-cm">${esc(m[1])}</span>`;
    else if(m[2]) out+=`<span class="k-str">${esc(m[2])}</span>`;
    else if(m[3]) out+=`<span class="k-num">${m[3]}</span>`;
    else if(m[4]) out+=PY_KW.has(m[4])?`<span class="k-kw">${m[4]}</span>`:esc(m[4]);
    else out+=esc(m[5]||m[6]||"");
  }
  return out;
}
function codeTabs(hostId, label, mathHtml, pyCode, symbols){
  const host=document.getElementById(hostId); if(!host) return;
  host.className="ct";
  const key = symbols && symbols.length
    ? `<details class="symkey"><summary>What every symbol means</summary>`+
      `<table class="symtab">`+
      symbols.map(([sym,mean])=>`<tr><td>${sym}</td><td>${mean}</td></tr>`).join("")+
      `</table></details>`
    : "";
  host.innerHTML=
    `<div class="ct-tabs"><button class="on" data-v="math">Math notation</button>`+
    `<button data-v="py">Pseudo-Python</button><span class="ct-label">${label}</span></div>`+
    `<div class="ct-body"><div class="ct-math mathblock">${mathHtml}${key}</div>`+
    `<pre class="code ct-py" style="display:none">${hlPython(pyCode)}</pre></div>`;
  host.querySelectorAll(".ct-tabs button").forEach(b=>b.onclick=()=>{
    host.querySelectorAll(".ct-tabs button").forEach(x=>x.classList.remove("on"));
    b.classList.add("on");
    host.querySelector(".ct-math").style.display = b.dataset.v==="math"?"":"none";
    host.querySelector(".ct-py").style.display   = b.dataset.v==="py"?"":"none";
  });
}

/* --- 1. emotion vector --- */
codeTabs("m1","emotion_vectors/extraction",
 `<span class="eq">
    <i>v</i><sub>e</sub><span class="op">=</span>
    <span class="frac">
      <span class="num">1</span>
      <span class="den"><span class="bigop"><span class="above"></span><span class="glyph">&sum;</span>
        <span class="below"><i>s</i></span></span><i>T</i><sub>s</sub></span>
    </span>
    <span class="bigop"><span class="above"></span><span class="glyph">&sum;</span>
      <span class="below"><i>s</i>&thinsp;&isin;&thinsp;<i>S</i><sub>e</sub></span></span>
    <i>T</i><sub>s</sub>
    <span class="frac"><span class="num">1</span><span class="den"><i>T</i><sub>s</sub></span></span>
    <span class="bigop"><span class="above"><i>n</i><sub>s</sub></span><span class="glyph">&sum;</span>
      <span class="below"><i>t</i>&thinsp;=&thinsp;51</span></span>
    <i>h</i><sub><i>t</i></sub><sup>(&ell;)</sup><span class="paren">(</span><i>s</i><span class="paren">)</span>
  </span>
  <span class="where">where <i>n</i><sub>s</sub> is the story's token count after padding is masked and
  <i>T</i><sub>s</sub> <span class="op">=</span> <i>n</i><sub>s</sub> <span class="op">&minus;</span> 50
  is how many tokens survive.</span>
  <span class="gl">The vector for emotion <i>e</i> is a mean of means. <b>Within a story</b> we average
  the residual-stream activation at layer &ell; over every token <i>except the first 50</i>, which the
  source paper treats as narrative framing. Padding is masked out too, and stories are capped at 512
  tokens. <b>Across stories</b> we take a token-weighted mean, so a long story counts for more than a
  short one. It is not read at a single token.</span>
  <span class="eq">
    <i>&#7805;</i><sub>e</sub>
    <span class="op">=</span> <i>v</i><sub>e</sub> <span class="op">&minus;</span>
    <span class="frac"><span class="num">1</span><span class="den"><i>E</i></span></span>
    <span class="bigop"><span class="above"><i>E</i></span><span class="glyph">&sum;</span>
      <span class="below"><i>e</i>&prime;&thinsp;=&thinsp;1</span></span>
    <i>v</i><sub><i>e</i>&prime;</sub>
  </span>
  <span class="gl">Then centred: subtract the mean over all <i>E</i> emotions, so what remains is what
  makes this emotion different rather than what all text shares. Every result in this deck uses
  <i>&#7805;</i>, never the raw <i>v</i>.</span>`,
`TOKEN_OFFSET = 50            # the source paper's convention: skip narrative framing
MAX_LENGTH   = 512

def pool_story(hidden, attention_mask):
    # mask padding AND the first 50 tokens, then mean over what survives
    mask = attention_mask.clone()
    mask[:TOKEN_OFFSET] = 0
    return (hidden * mask[:, None]).sum(0) / mask.sum().clamp(min=1)

def emotion_vector(stories, layer):
    # token-weighted across stories: a long story counts for more
    means, counts = [], []
    for s in stories:
        out = forward(s, max_length=MAX_LENGTH)
        means.append(pool_story(out.hidden[layer], out.attention_mask))
        counts.append(out.attention_mask.sum() - TOKEN_OFFSET)
    return sum(m * n for m, n in zip(means, counts)) / sum(counts)

def contrast_vectors(vectors):
    # centering: what is specific to each emotion, not shared by all text
    pool_mean = mean(vectors, axis=0)
    return vectors - pool_mean`,
 [["<i>e</i>","one of the 171 emotion words, for example <i>elated</i>"],
  ["<i>S</i><sub>e</sub>","the set of stories written to evoke emotion <i>e</i>"],
  ["<i>s</i>","one story in that set"],
  ["&ell;","the layer we read from (we captured 20 of them)"],
  ["<i>h</i><sub><i>t</i></sub><sup>(&ell;)</sup>","the model's residual-stream activation at token <i>t</i>, layer &ell;: one vector of 5,376 numbers"],
  ["<i>n</i><sub>s</sub>","how many real (non-padding) tokens story <i>s</i> has"],
  ["<i>T</i><sub>s</sub>","how many of those survive the pooling mask, that is <i>n</i><sub>s</sub> &minus; 50"],
  ["<i>v</i><sub>e</sub>","the raw emotion vector: the token-weighted mean over the whole set"],
  ["<i>&#7805;</i><sub>e</sub>","the <b>centred</b> vector, what we actually use everywhere"],
  ["<i>E</i>","how many emotions are in the pool being centred over (12 or 171)"]]
);

/* --- 2. PCA + the |r| = 0.83 claim --- */
codeTabs("m2","geometry_report/_context, _displacement",
 `<span class="eq">
    <i>X</i> <span class="op">&isin;</span> &#8477;<sup>171&times;<i>d</i></sup>
    <span class="op">,</span> <i>X</i> <span class="op">=</span>
    <i>U</i>&thinsp;<i>&Sigma;</i>&thinsp;<i>W</i><sup>&#8868;</sup>
  </span>
  <span class="gl">Stack the 171 centred emotion vectors as the rows of <i>X</i> and take its singular
  value decomposition. The principal components are the right singular vectors
  <i>w</i><sub>1</sub>&thinsp;&hellip;&thinsp;<i>w</i><sub><i>k</i></sub>, ordered by how much spread
  they explain. No human label enters this step.</span>
  <span class="eq">
    <i>z</i><sub><i>e</i>,<i>k</i></sub> <span class="op">=</span>
    <i>&#7805;</i><sub>e</sub>
    <span class="op">&middot;</span> <i>w</i><sub><i>k</i></sub>
  </span>
  <span class="gl">Each emotion's score on component <i>k</i>: how far along that direction it sits.</span>
  <span class="eq">
    <i>r</i><sub><i>k</i></sub> <span class="op">=</span>
    <span class="frac">
      <span class="num">
        <span class="bigop"><span class="above"></span><span class="glyph">&sum;</span>
          <span class="below"><i>e</i>&thinsp;&isin;&thinsp;<i>M</i></span></span>
        <span class="paren">(</span><i>z</i><sub><i>e</i>,<i>k</i></sub><span class="op">&minus;</span>
        <span class="bar"><i>z</i></span><span class="paren">)</span>
        <span class="paren">(</span><i>a</i><sub><i>e</i></sub><span class="op">&minus;</span>
        <span class="bar"><i>a</i></span><span class="paren">)</span>
      </span>
      <span class="den">
        &radic;<span style="border-top:1.1px solid currentColor;padding:0 .18em">
        <span class="bigop"><span class="above"></span><span class="glyph">&sum;</span>
          <span class="below"><i>e</i></span></span>
        <span class="paren">(</span><i>z</i><sub><i>e</i>,<i>k</i></sub><span class="op">&minus;</span>
        <span class="bar"><i>z</i></span><span class="paren">)</span><sup>2</sup>
        </span>
        &nbsp;&radic;<span style="border-top:1.1px solid currentColor;padding:0 .18em">
        <span class="bigop"><span class="above"></span><span class="glyph">&sum;</span>
          <span class="below"><i>e</i></span></span>
        <span class="paren">(</span><i>a</i><sub><i>e</i></sub><span class="op">&minus;</span>
        <span class="bar"><i>a</i></span><span class="paren">)</span><sup>2</sup>
        </span>
      </span>
    </span>
  </span>
  <span class="where">where <i>a</i><sub><i>e</i></sub> is the human valence rating of emotion word
  <i>e</i> in the NRC VAD lexicon, and <i>M</i> is the 164 of our 171 words that appear in it.</span>
  <span class="gl">Pearson correlation between the model's component scores and published human
  ratings: two independently produced quantities. We report |<i>r</i>| because the sign of a principal
  component is arbitrary. <b>Base model: |<i>r</i><sub>1</sub>| = 0.83. Instruct model: 0.11 at
  <i>k</i> = 1, and 0.72 at <i>k</i> = 3.</b></span>`,
`def pc_valence_correlation(vectors, layer, vad_lexicon):
    contrasts = contrast_vectors(vectors[:, layer, :])   # 171 x d, centered
    components, variance_ratio = pca(contrasts, n=5)     # SVD, model-only

    scores = contrasts @ components.T                    # 171 x 5
    matched = [e for e in emotions if e in vad_lexicon]  # 164 of 171
    human_valence = [vad_lexicon[e].valence for e in matched]

    out = []
    for k in range(5):
        model_scores = [scores[index_of(e), k] for e in matched]
        # sign of a principal component is arbitrary -> absolute value
        out.append(abs(pearson_r(model_scores, human_valence)))
    return out, variance_ratio`,
 [["<i>X</i>","the 171 centred emotion vectors stacked as rows"],
  ["<i>d</i>","the model's hidden width, 5,376 for Gemma 4 31B"],
  ["<i>U</i>, <i>&Sigma;</i>, <i>W</i>","the three factors of the singular value decomposition; the columns of <i>W</i> are the principal components"],
  ["<i>w</i><sub><i>k</i></sub>","the <i>k</i>-th principal component: a direction in activation space"],
  ["<i>k</i>","which component we mean; <i>k</i> = 1 is the largest"],
  ["<i>z</i><sub><i>e</i>,<i>k</i></sub>","emotion <i>e</i>'s score on component <i>k</i>: how far along that direction it sits"],
  ["<span class='bar'><i>z</i></span>","the mean score across the matched emotions"],
  ["<i>a</i><sub><i>e</i></sub>","the <b>human</b> valence rating of word <i>e</i> from the NRC VAD lexicon"],
  ["<span class='bar'><i>a</i></span>","the mean human rating across those same words"],
  ["<i>M</i>","the 164 of our 171 emotion words that appear in the lexicon"],
  ["<i>r</i><sub><i>k</i></sub>","Pearson correlation between the model's scores and the human ratings"]]
);

/* --- 3. tracking: gate rank and anticipation lead --- */
codeTabs("m3","q3_conventions.py, score_q3_gate_r1.py",
 `<span class="eq">
    <i>c</i><sub><i>t</i>,<i>p</i></sub> <span class="op">=</span>
    <span class="frac">
      <span class="num"><i>h</i><sub><i>t</i></sub> <span class="op">&middot;</span> <i>p</i>
        <span class="op">&minus;</span> <i>&mu;</i><sub><i>p</i></sub></span>
      <span class="den">&#8214; <i>h</i><sub><i>t</i></sub> <span class="op">&minus;</span>
        <i>&mu;</i> &#8214;</span>
    </span>
  </span>
  <span class="where">with <i>&mu;</i><sub><i>p</i></sub> the token-weighted mean of probe <i>p</i>'s
  dot products over the whole story set.</span>
  <span class="gl">The centred cosine between token <i>t</i> and probe <i>p</i>. Centring on the
  <b>story-set</b> mean is what stops a probe that is simply large everywhere from winning by
  default.</span>
  <span class="eq">
    rank<sub><i>&phi;</i></sub> <span class="op">=</span> 1 <span class="op">+</span>
    <span class="paren">|</span>{ <i>p</i> <span class="op">:</span>
    <span class="bar"><i>c</i></span><sub><i>&phi;</i>,<i>p</i></sub>
    <span class="op">&gt;</span>
    <span class="bar"><i>c</i></span><sub><i>&phi;</i>,<i>e</i>(<i>&phi;</i>)</sub>
    }<span class="paren">|</span>
    <span class="op">,</span>
    <span class="bar"><i>c</i></span><sub><i>&phi;</i>,<i>p</i></sub>
    <span class="op">=</span>
    <span class="frac"><span class="num">1</span><span class="den">|<i>&phi;</i>|</span></span>
    <span class="bigop"><span class="above"></span><span class="glyph">&sum;</span>
      <span class="below"><i>t</i>&thinsp;&isin;&thinsp;<i>&phi;</i></span></span>
    <i>c</i><sub><i>t</i>,<i>p</i></sub>
  </span>
  <span class="gl"><b>Naming the current emotion.</b> The model reads the story once and we keep
  <b>every token</b>. We compute the cosine at each position, then average it over the tokens
  belonging to phase <i>&phi;</i>. Phases shorter than 4 tokens are too short to average, so we
  skip them. Then we count how many of the twelve emotion vectors beat the tagged one. Rank 1 is perfect,
  and with twelve to choose from, guessing gives 6.5.</span>
  <span class="eq">
    lead<sub><i>b</i></sub> <span class="op">=</span>
    <span class="frac"><span class="num">1</span><span class="den"><i>W</i></span></span>
    <span class="bigop"><span class="above"><i>b</i>&minus;1</span><span class="glyph">&sum;</span>
      <span class="below"><i>t</i>&thinsp;=&thinsp;<i>b</i>&minus;<i>W</i></span></span>
    <i>c</i><sub><i>t</i>,<i>q</i></sub> <span class="op">&minus;</span>
    <span class="frac"><span class="num">1</span><span class="den"><i>W</i></span></span>
    <span class="bigop"><span class="above"><i>b</i>&minus;<i>W</i>&minus;1</span>
      <span class="glyph">&sum;</span>
      <span class="below"><i>t</i>&thinsp;=&thinsp;<i>b</i>&minus;2<i>W</i></span></span>
    <i>c</i><sub><i>t</i>,<i>q</i></sub>
  </span>
  <span class="where">at a written turn <i>b</i>, for the incoming emotion's probe <i>q</i>, with
  <i>W</i> = 16 tokens.</span>
  <span class="gl"><b>Anticipation.</b> The mean cosine in the 16 tokens before the turn minus the mean
  in the 16 before those. Above zero means the next emotion is already rising before the story turns.
  Both windows are referenced to the boundary, so story length cannot drive the effect.</span>`,
`W = 16                       # window size, fixed before any scoring
LAYERS = [6, 15, 24, 33, 42, 51]
MIN_PHASE_TOKENS = 4         # shorter phases are skipped, not averaged

# NOTE the difference from probe extraction: there we pooled ONE vector per
# story (masked mean over tokens 50..end). Here we keep the per-token series,
# because the question is how the reading MOVES through the story.

def centered_cos(shard, story_set_mean):
    # story_set_mean is token-weighted over the whole corpus, not per story
    return (shard["dots"] - story_set_mean) / shard["norms_centered"][:, :, None]

def rank_of_tagged_emotion(cos, phase, tagged_probe, emotion_set):
    phase_mean = cos[phase.start:phase.end, :, emotion_set].mean(axis=0)
    beaten_by = (phase_mean > phase_mean[tagged_probe]).sum()
    return beaten_by + 1          # 1 = the tagged probe wins outright

def anticipation_lead(cos, boundary, incoming_probe):
    near    = cos[boundary - W:boundary,       :, incoming_probe].mean(axis=0)
    earlier = cos[boundary - 2*W:boundary - W, :, incoming_probe].mean(axis=0)
    return near - earlier          # > 0 = the next emotion is already rising`,
 [["<i>t</i>","a token position in the story"],
  ["<i>p</i>","one probe, that is one emotion's centred vector"],
  ["<i>h</i><sub><i>t</i></sub>","the model's activation while reading token <i>t</i>"],
  ["<i>&mu;</i><sub><i>p</i></sub>","probe <i>p</i>'s mean dot product across the whole story set (the centring term)"],
  ["&#8214;&thinsp;&middot;&thinsp;&#8214;","vector length, so dividing by it turns a dot product into a cosine"],
  ["<i>c</i><sub><i>t</i>,<i>p</i></sub>","the centred cosine: how close token <i>t</i> reads to probe <i>p</i>"],
  ["<i>&phi;</i>","one phase of the story, that is one tagged emotion's stretch of tokens"],
  ["|<i>&phi;</i>|","how many tokens that phase has (fewer than 4 and we skip it)"],
  ["<i>e</i>(<i>&phi;</i>)","the emotion phase <i>&phi;</i> was written to express"],
  ["<span class='bar'><i>c</i></span><sub><i>&phi;</i>,<i>p</i></sub>","probe <i>p</i>'s average cosine over that phase's tokens"],
  ["rank<sub><i>&phi;</i></sub>","where the tagged emotion places among the twelve; 1 is best"],
  ["<i>b</i>","the token index where the story is written to turn"],
  ["<i>W</i>","the window length, fixed at 16 tokens before scoring"],
  ["<i>q</i>","the probe for the <b>incoming</b> emotion, the one after the turn"],
  ["lead<sub><i>b</i></sub>","how much the incoming emotion rises just before the turn"]]
);

/* --- 4. the nulls --- */
codeTabs("m4","score_q3_gate_r1.py (N1, N2)",
 `<span class="eq">
    <i>p</i> <span class="op">=</span>
    <span class="frac"><span class="num">1</span><span class="den"><i>B</i></span></span>
    <span class="bigop"><span class="above"><i>B</i></span><span class="glyph">&sum;</span>
      <span class="below"><i>b</i>&thinsp;=&thinsp;1</span></span>
    <b>1</b><span class="paren">[</span>
    median<span class="paren">(</span>rank<sup>(<i>b</i>)</sup><span class="paren">)</span>
    <span class="op">&le;</span>
    median<span class="paren">(</span>rank<sup>obs</sup><span class="paren">)</span>
    <span class="paren">]</span>
  </span>
  <span class="where">over <i>B</i> = 10,000 shuffles in which every phase is re-scored against a
  randomly assigned <b>wrong</b> emotion.</span>
  <span class="gl"><b>N2, the wrong-emotion shuffle.</b> How often does chance beat what we observed?
  This shuffle sets the floor the real result has to beat.</span>
  <span class="eq">
    <i>&#7805;</i><sub>rand</sub>
    <span class="op">&sim;</span> span
    <span class="paren">{</span>
    <i>&#7805;</i><sub>e</sub>
    <span class="paren">}</span>
    <span class="op">,</span>
    lead <span class="op">&ge;</span>
    <span class="frac"><span class="num">1</span><span class="den">2</span></span>
    <i>&sigma;</i><sub>noise</sub><span class="paren">(</span>&ell;<span class="paren">)</span>
  </span>
  <span class="gl"><b>N1, the random-direction control.</b> Directions drawn from the span of the probe
  set, used to calibrate a per-layer noise scale. An effect must clear half that noise standard
  deviation, so a tiny but consistent drift cannot pass on a <i>p</i>-value alone. A claim graduates
  only when <b>both</b> conditions hold.</span>`,
`B = 10_000                    # shuffles, count fixed in advance

def wrong_emotion_null(cos, phases, emotion_set, rng):
    # rank EVERY probe once, then index the shuffled picks: 10k shuffles
    # without 10k re-sorts
    order = argsort(-phase_means, axis=1)
    rank_of_probe = argsort(order, axis=1) + 1

    null_medians = []
    for _ in range(B):
        wrong = rng.choice([p for p in emotion_set if p != tagged], size=len(phases))
        null_medians.append(median(take_along(rank_of_probe, wrong)))
    return mean(null_medians <= observed_median)      # the p-value

# a claim graduates only if BOTH hold:
#   p < 0.001  and  effect >= 0.5 * calibrated_noise_sd[layer]`,
 [["<i>B</i>","how many shuffles we run, fixed at 10,000"],
  ["<i>b</i>","one of those shuffles"],
  ["rank<sup>(<i>b</i>)</sup>","the ranks obtained in shuffle <i>b</i>, where every phase was scored against a randomly picked <b>wrong</b> emotion"],
  ["rank<sup>obs</sup>","the ranks we actually observed"],
  ["<b>1</b>[&thinsp;&middot;&thinsp;]","the indicator: 1 when the statement inside is true, 0 otherwise"],
  ["<i>p</i>","the resulting p-value: the share of shuffles that did at least as well as we did"],
  ["<i>&#7805;</i><sub>rand</sub>","a random direction drawn from the span of the real probes"],
  ["<i>&sigma;</i><sub>noise</sub>(&ell;)","the spread those random directions produce at layer &ell;: our noise scale"]]
);

/* ---------- section index on the cover ---------- */
/* Walked in document order so each section picks up the part divider it falls
   under. The parts used to exist only as a bar in the body: the contents and
   the tick row were both built from section[id] alone, so a reader met "PART
   TWO" while both navigation aids showed a flat list of eleven. One structure
   now, built once. */
const SECTIONS=[];
{
  let part=null;
  document.querySelectorAll("section[id], .partbar[data-part]").forEach(node=>{
    if(node.classList.contains("partbar")){
      part={no:node.dataset.partNo, title:node.dataset.part};
      return;
    }
    // the parts cover the argument, not the back matter: next steps, methods
    // and the glossary belong to none of them
    if(node.hasAttribute("data-endparts")) part=null;
    SECTIONS.push({id:node.id, n:SECTIONS.length+1, part,
      title:node.querySelector("h2").textContent.replace(/^\s*\d+\s*/,"").trim()});
  });
}
document.getElementById("coverIndex").innerHTML = SECTIONS.map((x,i)=>{
  const opens = x.part && (i===0 || SECTIONS[i-1].part!==x.part);
  return (opens ? `<span class="ipart"><b>Part ${x.part.no}</b> &middot; ${x.part.title}</span>` : "")
    + `<a href="#${x.id}"><span class="n">${x.n}</span><span class="t">${x.title}</span></a>`;
}).join("");

/* ---------- keyboard: next / previous / jump ---------- */
/* Track the section explicitly rather than re-deriving it from scrollY: a
   smooth scroll is still animating when the next key arrives, so deriving
   would make two quick presses land on the same section. Manual scrolling
   re-syncs the index. */
let navIdx = -1, navAnimating = false, scrollEndTimer = null;
function goToSection(i){
  navIdx = Math.max(0, Math.min(SECTIONS.length-1, i));
  navAnimating = true;   // cleared when scrolling actually stops, not on a guessed timeout
  document.getElementById(SECTIONS[navIdx].id).scrollIntoView({behavior:"smooth",block:"start"});
}
function sectionFromScroll(){
  const y=scrollY+140; let idx=-1;
  SECTIONS.forEach((x,i)=>{if(document.getElementById(x.id).offsetTop<=y) idx=i;});
  return idx;
}
// Sync the index only once scrolling has settled. Syncing during a smooth
// scroll would overwrite the target the user just asked for with wherever the
// animation happens to be, which made two fast presses land on one section.
addEventListener("scroll",()=>{
  clearTimeout(scrollEndTimer);
  scrollEndTimer = setTimeout(()=>{
    if(navAnimating) navAnimating = false;   // our own scroll finished; keep navIdx
    else navIdx = sectionFromScroll();       // the user scrolled by hand
  }, 130);
},{passive:true});
addEventListener("keydown",e=>{
  // never hijack typing, and leave modified keys to the browser
  const tag=(e.target.tagName||"").toLowerCase();
  // A focused control owns its own keys. Space toggles a button and opens a
  // <summary>; hijacking it for section-advance broke every one of them.
  if(["input","textarea","button","summary","select","a"].includes(tag)
     || e.target.isContentEditable || e.metaKey||e.ctrlKey||e.altKey) return;
  if(e.key==="ArrowRight"||e.key===" "||e.key==="PageDown"){e.preventDefault();goToSection(navIdx+1);}
  else if(e.key==="ArrowLeft"||e.key==="PageUp"){e.preventDefault();goToSection(navIdx-1);}
  else if(e.key==="Home"){e.preventDefault();navIdx=-1;scrollTo({top:0,behavior:"smooth"});}
  else if(/^[0-9]$/.test(e.key)){
    // 1-9 are sections 1-9; 0 is section 10, the Q&A
    const n=e.key==="0"?10:parseInt(e.key,10);
    if(n<=SECTIONS.length){e.preventDefault();goToSection(n-1);}
  }
});

/* ---------- cross-layer RSA matrices ---------- */
const RSA_KEYS=Object.keys(D.rsa);
/* Button label AND figure title come from here. The raw keys are the notebook's
   own names and carry jargon ("unablated", "RSA"); they never reach the page. */
const RSA_LABEL={"instruct RSA (unablated)":"instruction-tuned model",
  "base RSA (unablated)":"base model",
  "instruct RSA (top component removed)":"instruction-tuned, top PC removed",
  "cross-model RSA: instruct vs base":"base vs. instruction-tuned"};
/* the one-line reminder under the buttons: what THIS view is for */
const RSA_NOTE={"instruct RSA (unablated)":"do the instruction-tuned model's layers agree with each other?",
  "base RSA (unablated)":"the control: the same question on the base model",
  "instruct RSA (top component removed)":"the same layers, with only the mystery axis removed",
  "cross-model RSA: instruct vs base":"one model on each axis: what did instruction tuning change?"};
let rsaKey=RSA_KEYS[0];
function drawRsa(){
  const host=document.getElementById("rsaChart"); if(!host) return;
  host.innerHTML="";
  const m=D.rsa[rsaKey], z=m.z, layers=m.layers||z.map((_,i)=>i);
  const n=z.length, W=760,H=452,L=62,T=36,B=52,Rr=150;
  // Only the cross-model view has a different model on each axis, and it is the
  // one view that is NOT symmetric — so orientation has to be on the figure.
  const isCross=rsaKey.indexOf("cross-model")===0;
  const rowName=isCross?"layer of the instruction-tuned model"
                       :"layer of the model (same model on both axes)";
  const colName=isCross?"layer of the base model"
                       :"layer of the model (same model on both axes)";
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const size=Math.min((W-L-Rr)/n,(H-T-B)/n);
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){
    const v=z[i][j];
    const cell=el("rect",{x:L+j*size,y:T+i*size,width:size+.5,height:size+.5,
      fill:`rgba(29,53,87,${Math.max(0,Math.min(1,v)).toFixed(3)})`});
    tipOn(cell,`<b>${isCross?"instruction-tuned":""} layer ${layers[i]} vs ${isCross?"base":""} `+
      `layer ${layers[j]}</b>: agreement ${v.toFixed(2)}`+
      `<span class="t-sub">1 means these two layers sort the 171 emotions the same way; `+
      `0 means they disagree completely.</span>`);
    svg.appendChild(cell);
  }
  [0,Math.floor(n/2),n-1].forEach(i=>{
    const a=el("text",{x:L-6,y:T+i*size+size/2+4,"text-anchor":"end","font-size":10,fill:P.muted});
    a.textContent="L"+layers[i]; svg.appendChild(a);
    const b=el("text",{x:L+i*size+size/2,y:T+n*size+16,"text-anchor":"middle","font-size":10,fill:P.muted});
    b.textContent="L"+layers[i]; svg.appendChild(b);
  });
  // the scale, with both ends named
  const sx=L+n*size+26, sy=T, sh=n*size;
  for(let i=0;i<40;i++)
    svg.appendChild(el("rect",{x:sx,y:sy+sh-(i+1)*(sh/40),width:13,height:sh/40+.5,
      fill:`rgba(29,53,87,${((i/39)).toFixed(3)})`}));
  [["1.0  sort emotions identically",sy+6],
   ["0.5  partly agree",sy+sh/2],
   ["0.0  no agreement at all",sy+sh]].forEach(([t,y])=>{
    const tx=el("text",{x:sx+18,y:y+4,"font-size":10,fill:P.muted}); tx.textContent=t; svg.appendChild(tx);
  });
  const xl=el("text",{x:L+(n*size)/2,y:H-8,"text-anchor":"middle","font-size":10.5,fill:P.muted});
  xl.textContent=colName; svg.appendChild(xl);
  const ylab=el("text",{x:14,y:T+(n*size)/2,"font-size":10.5,fill:P.muted,
    transform:`rotate(-90 14 ${T+(n*size)/2})`,"text-anchor":"middle"});
  ylab.textContent=rowName; svg.appendChild(ylab);
  // the title lives INSIDE the svg: an exported png of this chart is otherwise
  // three near-identical matrices with nothing saying which one you are seeing
  const ttl=el("text",{x:L,y:20,"font-size":12.5,fill:P.text,"font-weight":600});
  ttl.textContent=RSA_LABEL[rsaKey]||rsaKey; svg.appendChild(ttl);
  host.appendChild(svg);
  document.getElementById("rsaNote").textContent=RSA_NOTE[rsaKey]||"";
}
(function(){
  const host=document.getElementById("rsaBtns"); if(!host) return;
  RSA_KEYS.forEach((k,i)=>{
    const b=document.createElement("button");
    b.className="seg"+(i===0?" on":""); b.textContent=RSA_LABEL[k]||k;
    b.onclick=()=>{host.querySelectorAll("button").forEach(x=>x.classList.remove("on"));
      b.classList.add("on"); rsaKey=k; drawRsa();};
    host.appendChild(b);
  });
})();

/* ---------- progress: the nav is the tracker, no extra surface ---------- */
/* The ticks are grouped the way the contents is: a hairline opens each part, so
   the bar shows the three-beat shape at a glance without spelling anything out.
   The part name rides in the title of its first tick, where a reader who wants
   it can find it. */
document.getElementById("navTicks").innerHTML = SECTIONS.map((x,i)=>{
  // a separator wherever the part changes, including where the parts end and
  // the back matter begins
  const opens = i>0 && SECTIONS[i-1].part!==x.part;
  const label = (x.part?`Part ${x.part.no}, ${x.part.title}: `:"")
    + `${x.n}. ${x.title}`;
  return `<a href="#${x.id}" class="${opens?"partstart":""}" `
    + `title="${label.replace(/"/g,"&quot;")}">${x.n}</a>`;
}).join("");
const NAV_LINKS=[...document.querySelectorAll("nav .ticks a")];
const INDEX_LINKS=[...document.querySelectorAll("#coverIndex a")];
const HERE=document.getElementById("navHere");
function paintProgress(){
  const cur=sectionFromScroll();
  NAV_LINKS.forEach((a,i)=>{
    a.classList.toggle("on", i===cur);
    a.classList.toggle("done", i<cur);
  });
  // the contents marks the same section the tick bar does
  INDEX_LINKS.forEach((a,i)=>a.classList.toggle("on", i===cur));
  HERE.textContent = cur<0 ? "" : SECTIONS[cur].title;
  const max=document.body.scrollHeight-innerHeight;
  document.getElementById("bar").style.width=(max>0?(scrollY/max)*100:0)+"%";
}
addEventListener("scroll",paintProgress,{passive:true});
paintProgress();

/* ---------- expand any figure ---------- */
const MODAL=document.getElementById("modal"), MODAL_BODY=document.getElementById("modalBody");
function closeModal(){
  MODAL.classList.remove("on"); MODAL_BODY.innerHTML="";
  if(MODAL_OPENER){ MODAL_OPENER.focus(); MODAL_OPENER=null; }
}
const MODAL_CLOSE=document.getElementById("modalClose");
let MODAL_OPENER=null;
MODAL_CLOSE.onclick=closeModal;
// the same treatment as the expand control it mirrors; a bare glyph with no
// hover text is a guess for the reader
tipOn(MODAL_CLOSE,"Close this figure");
MODAL_CLOSE.style.cursor="pointer";
MODAL.onclick=e=>{if(e.target===MODAL) closeModal();};
addEventListener("keydown",e=>{if(e.key==="Escape") closeModal();});
// the element holding each chart's live state, to carry into the expanded view
const FIG_NOTE={pcChart:"pcVerdict", gridChart:"gridNote", lineChart:"tokLabel",
  ternChart:"tokLabel", emoChart:"emoNote", rsaChart:"rsaNote"};
function addModalLine(text,cls,style){
  const d=document.createElement("div");
  d.className=cls||"muted";
  d.style.cssText=style||"font-size:13px;margin:0 0 12px";
  d.textContent=text; MODAL_BODY.appendChild(d);
}
// wrap every chart host so it gets an expand button; the modal shows a live clone
/* Driven off data-figtitle rather than a hand-kept list of ids, which had
   drifted: the two cover charts and the method diagram carried no expand
   control at all. Anything that declares itself a figure now gets one. */
[...document.querySelectorAll("[data-figtitle]")].forEach(host=>{
  const id=host.id;
  const parent=host.parentElement;
  if(!parent.classList.contains("figwrap")) parent.classList.add("figwrap");
  if(parent.querySelector(".expand")) return;
  const b=document.createElement("button");
  b.className="expand"; b.setAttribute("aria-label","Expand this figure");
  // the page's own tooltip, not the browser's: it appears at once and matches
  // every other hover here, where title= waits a second and looks foreign
  tipOn(b,"Expand this figure");
  b.style.cursor="pointer";
  b.innerHTML='<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M9 3H4a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V7"/><path d="M10.5 2h3.5v3.5"/><path d="M14 2l-5.5 5.5"/></svg>';
  b.onclick=()=>{
    const svg=host.querySelector("svg"); if(!svg) return;
    MODAL_BODY.innerHTML="";
    // Everything that explains a figure lives OUTSIDE its svg: the heading and
    // source line in the enclosing card, the caption in a sibling div, the live
    // state in the controls row. Cloning the svg alone produced an expanded
    // figure with no title, which is the least readable view of all.
    const card=host.closest(".card") || parent;
    // the figure names itself via data-figtitle, so the expanded view never
    // depends on whatever markup happens to surround the chart
    const named=host.dataset.figtitle || (card.querySelector("h3")||{}).textContent;
    // carry the number across, so a reader who expands figure 7 still knows
    // which figure they are looking at when they come back to the page
    const title=named && host.dataset.fignum ? `Figure ${host.dataset.fignum}. ${named}` : named;
    if(title) addModalLine(title,"",
      "font-family:var(--display);font-size:17px;font-weight:600;color:var(--text);margin:0 0 4px");
    const cap=host.previousElementSibling;
    if(cap && cap.classList && cap.classList.contains("figcap"))
      addModalLine(cap.textContent,"muted","font-size:13px;margin:0 0 10px");
    const note=FIG_NOTE[id] && document.getElementById(FIG_NOTE[id]);
    if(note && note.textContent.trim())
      addModalLine(note.textContent,"muted","font-size:13px;margin:0 0 10px");
    MODAL_BODY.appendChild(svg.cloneNode(true));
    // The legend lives beside the chart, not inside the svg, so a clone of the
    // svg alone arrives with every colour unexplained.
    const legend=card.querySelector(".legend");
    if(legend){
      const l=document.createElement("div");
      l.className="legend"; l.innerHTML=legend.innerHTML;
      MODAL_BODY.appendChild(l);
    }
    // the how-to-read block and the evidence file, both of which the acceptance
    // test for these figures depends on
    const howto=card.querySelector("details.howto");
    if(howto){
      const d=document.createElement("details");
      d.className="howto"; d.style.marginTop="14px";
      d.innerHTML=howto.innerHTML;
      MODAL_BODY.appendChild(d);
    }
    const src=card.querySelector(".src");
    if(src) addModalLine(src.textContent,"src","font-size:11px;margin-top:10px");
    MODAL.classList.add("on");
    // Focus moves into the dialog, and comes back to the control that opened it
    // when it closes. Without this the keyboard stays behind the overlay.
    MODAL_OPENER=b;
    // #modal is display:none until .on, and focus() on a display:none element
    // is silently dropped, so the focus move waits for the next frame.
    requestAnimationFrame(()=>MODAL_CLOSE.focus());
  };
  parent.appendChild(b);
});

/* Every figure gets a visible number and title above it. Both already existed
   in data-figtitle, but only the expand-to-full-screen view ever read them, so
   on the page itself a figure arrived unnamed. Numbering is by document order
   rather than hand-written, so inserting a figure renumbers the rest instead of
   leaving a duplicate "Figure 4" behind. */
function numberFigures(){
  document.querySelectorAll("[data-figtitle]").forEach((host,i)=>{
    const n=i+1;
    host.dataset.fignum=n;                       // the modal reads this back
    const cap=el2("div",{class:"figtitle"});
    const tag=el2("span",{class:"n"}); tag.textContent="Figure "+n;
    const ttl=el2("span",{class:"t"}); ttl.textContent=host.dataset.figtitle;
    cap.appendChild(tag); cap.appendChild(ttl);
    // Section 6 already puts a small .figcap sub-label directly above its
    // charts, and the expand view finds it as the host's previous sibling.
    // Going in above that label keeps both the lookup and the reading order.
    const prev=host.previousElementSibling;
    const anchor=(prev && prev.classList && prev.classList.contains("figcap")) ? prev : host;
    anchor.parentNode.insertBefore(cap,anchor);
  });
}
/* plain-HTML sibling of el(), which builds SVG-namespaced nodes */
function el2(t,a={}){const e=document.createElement(t);for(const k in a)e.setAttribute(k,a[k]);return e;}

wireGlossary();
openTargetedEntry();   // a cold load of #g-elo lands on an open entry, not a shut one
numberFigures();
drawPCs("base","coverPcBase",null);
drawPCs("instruct","coverPcIt",null);
drawPrefLayers(); drawSteering();
drawPCs("base"); drawGrid(); drawStory(); drawEmo(); drawLayers(); drawLineage(); drawDose(); drawRsa();
