const API_KEY = "blank_for_demo_purposes_only"; 

const LAT = 51.5074;
const LON = -0.1278;

let predictedData = []; // [{ time: "14:00", carbon: 150, ts: Date }, ...]
let chartInstance = null;

async function init() {
    try {
        window.session = await ort.InferenceSession.create('./grid_model.onnx');
        
        navigator.geolocation.getCurrentPosition(
            (pos) => fetchAndPredict(pos.coords.latitude, pos.coords.longitude),
            () => fetchAndPredict(LAT, LON)
        );
        
        setupEventListeners();

    } catch (e) {
        console.error("Initialization Error:", e);
        alert("Error loading model. Make sure grid_model.onnx is in the folder.");
    }
}

function setupEventListeners() {
    const btnSchedule = document.getElementById('btn-schedule');
    btnSchedule.addEventListener('click', handleScheduleClick);
}

async function fetchAndPredict(lat, lon) {
    document.getElementById('status-badge').innerText = `LOC: ${lat.toFixed(2)}, ${lon.toFixed(2)}`;

    const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&hourly=temperature_2m,shortwave_radiation,wind_speed_10m&past_days=0&forecast_days=2`;
    const res = await fetch(url);
    const weather = await res.json();

    const batchSize = 24;
    const inputSize = 6;
    const inputData = new Float32Array(batchSize * inputSize);
    const nowHour = new Date().getHours();
    
    predictedData = [];

    for (let i = 0; i < batchSize; i++) {
        let idx = nowHour + i;
        if (!weather.hourly.time[idx]) break;

        let date = new Date(weather.hourly.time[idx]);
        
        let offset = i * inputSize;
        inputData[offset+0] = date.getHours();
        inputData[offset+1] = date.getMonth() + 1;
        inputData[offset+2] = date.getDay();
        inputData[offset+3] = weather.hourly.temperature_2m[idx];
        inputData[offset+4] = weather.hourly.shortwave_radiation[idx];
        inputData[offset+5] = weather.hourly.wind_speed_10m[idx];
        
        predictedData.push({
            time: date.getHours() + ":00",
            ts: date,
            carbon: 0 
        });
    }

    const tensor = new ort.Tensor('float32', inputData, [batchSize, inputSize]);
    const feeds = { float_input: tensor }; 
    const results = await window.session.run(feeds);
    const output = results.variable.data;

    for(let i=0; i<predictedData.length; i++) {
        predictedData[i].carbon = Math.max(0, Math.round(output[i]));
    }

    renderDashboard();
}

async function getApplianceData(taskName) {
    console.warn("Gemini API failed or key missing, using fallback.");
    return { power_kw: 1.0, type: "attended" };
}

function findBestSlot(durationHrs, type) {
    const GREEN_THRESHOLD = 180;
    let bestIdx = 0;
    let minCarbon = Infinity;
    let worstCarbon = 0;

    const nowHour = new Date().getHours();
    
    for (let i = 0; i <= 12; i++) {
        if (i + durationHrs >= predictedData.length) break;

        const startBlock = predictedData[i];
        const startHour = startBlock.ts.getHours();

        if (type === 'attended') {
            if (startHour < 7 || startHour > 22) continue; 
        }

        let totalWindowCarbon = 0;
        for (let j = 0; j < durationHrs; j++) {
            totalWindowCarbon += predictedData[i+j].carbon;
        }

        if (totalWindowCarbon < minCarbon) {
            minCarbon = totalWindowCarbon;
            bestIdx = i;
        }
        if (totalWindowCarbon > worstCarbon) {
            worstCarbon = totalWindowCarbon;
        }
    }

    const currentCarbon = predictedData[0].carbon;
    const bestStartCarbon = predictedData[bestIdx].carbon;
    const savedPercent = Math.round(((worstCarbon - minCarbon) / worstCarbon) * 100);
    
    let status = "run_now";
    if (bestIdx > 0 && (currentCarbon - bestStartCarbon > 30)) status = "wait"; 
    if (minCarbon > (GREEN_THRESHOLD * durationHrs)) status = "dirty_all_day";

    return {
        startTime: predictedData[bestIdx].time,
        startObj: predictedData[bestIdx],
        savedPercent: savedPercent,
        status: status,
        avgCarbon: Math.round(minCarbon / durationHrs)
    };
}

async function handleScheduleClick() {
    const btn = document.getElementById('btn-schedule');
    const customInput = document.getElementById('custom-task-name');
    const durationInput = document.getElementById('task-duration');
    const emptyState = document.getElementById('empty-state');
    const activeBtn = document.querySelector('.task-btn.border-white'); // currently selected

    if (!activeBtn) {
        alert("Please select a task first.");
        return;
    }

    const originalText = btn.innerHTML;
    btn.innerHTML = `<div class="w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin"></div> Determining Power...`;
    btn.disabled = true;

    let taskName = activeBtn.getAttribute('data-task');
    if (taskName === 'Other') taskName = customInput.value || "Custom Appliance";
    
    const appData = await getApplianceData(taskName);
    
    const duration = parseFloat(durationInput.value);
    const result = findBestSlot(duration, appData.type);
    
    if(emptyState) emptyState.style.display = 'none';
    
    const totalImpact = (appData.power_kw * duration * (result.avgCarbon / 1000)).toFixed(2); // kgCO2
    
    let colorClass = "bg-gray-800 border-gray-700";
    let icon = "⚡";
    let message = `Start at ${result.startTime}`;
    
    if (result.status === "wait") {
        colorClass = "bg-yellow-900/30 border-yellow-700/50";
        message = `Wait until ${result.startTime} to save ${result.savedPercent}%`;
    } else if (result.status === "dirty_all_day") {
        colorClass = "bg-red-900/20 border-red-800/50";
        icon = "⚠️";
        message = "High Grid Impact Today";
    } else {
        colorClass = "bg-green-900/20 border-green-800/50";
        message = "Grid is Clean - Run Now";
    }

    const cardHTML = `
        <div class="${colorClass} border p-4 rounded-lg flex flex-col gap-3 animate-fade-in relative overflow-hidden group">
            <div class="flex justify-between items-start z-10">
                <div class="flex items-center gap-3">
                    <div class="bg-black/40 p-2 rounded text-xl">${icon}</div>
                    <div>
                        <h4 class="font-bold text-white text-sm leading-tight">${taskName}</h4>
                        <p class="text-[10px] text-gray-400 uppercase tracking-wide mt-0.5">
                            ${duration}h
                        </p>
                    </div>
                </div>
                <div class="text-right">
                    <span class="block text-xs font-bold text-white">${message}</span>
                </div>
            </div>
            
            <!-- Remove Button -->
            <button onclick="this.parentElement.remove()" class="absolute top-2 right-2 text-gray-600 hover:text-red-500 opacity-0 group-hover:opacity-100 transition">✕</button>
        </div>
    `;

    document.getElementById('task-list').insertAdjacentHTML('afterbegin', cardHTML);

    // Reset
    btn.innerHTML = originalText;
    btn.disabled = false;
}

// RENDERING & HELPERS
function renderDashboard() {
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('dashboard').classList.remove('hidden');

    const current = predictedData[0];
    const isClean = current.carbon < 180;

    const heroTitle = document.getElementById('hero-title');
    heroTitle.innerText = isClean ? "LOW" : "HIGH";
    heroTitle.className = `text-7xl font-black tracking-tighter ${isClean ? "text-green-500" : "text-red-500"}`;
    document.getElementById('hero-badge').innerText = `${current.carbon} gCO2/kWh`;
    
    document.getElementById('rec-time').innerText = isClean ? "Good Condition" : "Grid Strained";
    document.getElementById('rec-desc').innerText = isClean 
        ? "Renewable generation is currently high." 
        : "High demand is causing fossil fuel usage.";

    renderChart();
}

function renderChart() {
    const ctx = document.getElementById('mainChart').getContext('2d');
    if(chartInstance) chartInstance.destroy();

    const colors = predictedData.map(d => d.carbon < 180 ? '#22c55e' : '#ef4444');

    chartInstance = new Chart(ctx, {
        type: 'bar', // Bar chart often looks better for hourly slots
        data: {
            labels: predictedData.map(d => d.time),
            datasets: [{
                data: predictedData.map(d => d.carbon),
                backgroundColor: colors,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { display: false }, ticks: { color: '#666', font: {size: 10} } },
                y: { display: false }
            },
            plugins: { legend: { display: false } }
        }
    });
}

init();
