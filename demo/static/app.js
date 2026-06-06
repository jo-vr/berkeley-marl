const conditionSelect = document.querySelector('#condition');
const resetButton = document.querySelector('#reset');
const stepButton = document.querySelector('#step');
const playButton = document.querySelector('#play');
const pauseButton = document.querySelector('#pause');
const grid = document.querySelector('#grid');
const stepCount = document.querySelector('#stepCount');
const reward = document.querySelector('#reward');
const done = document.querySelector('#done');
const success = document.querySelector('#success');
const actionsEl = document.querySelector('#actions');

let timer = null;
let lastState = null;

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function manhattan(a, b) {
  return Math.abs(a.row - b.row) + Math.abs(a.col - b.col);
}

function render(payload) {
  const state = payload.state;
  lastState = state;
  grid.innerHTML = '';
  grid.style.gridTemplateColumns = `repeat(${state.grid_size}, 1fr)`;

  const agentsByCell = new Map();
  for (const [name, agent] of Object.entries(state.agents)) {
    agentsByCell.set(`${agent.row},${agent.col}`, name.replace('agent_', 'A'));
  }

  for (let row = 0; row < state.grid_size; row++) {
    for (let col = 0; col < state.grid_size; col++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      const key = `${row},${col}`;
      const isTree = row === state.tree.row && col === state.tree.col;
      if (isTree) cell.classList.add('tree');
      if (agentsByCell.has(key)) {
        cell.classList.add('agent');
        cell.dataset.agent = agentsByCell.get(key);
      }
      for (const agent of Object.values(state.agents)) {
        if (manhattan({row, col}, agent) <= 1) cell.classList.add('visible');
      }
      grid.appendChild(cell);
    }
  }

  stepCount.textContent = state.step;
  reward.textContent = payload.reward ?? '0';
  done.textContent = payload.done ?? state.done;
  success.textContent = payload.success ?? false;
  actionsEl.textContent = JSON.stringify(payload.actions ?? {});
}

async function reset() {
  const payload = await postJson('/api/reset', {condition: conditionSelect.value});
  render(payload);
}

async function step() {
  const payload = await postJson('/api/step', {condition: conditionSelect.value});
  render(payload);
  if (payload.done && timer) pause();
}

function play() {
  if (!timer) timer = setInterval(step, 500);
}

function pause() {
  if (timer) clearInterval(timer);
  timer = null;
}

conditionSelect.addEventListener('change', reset);
resetButton.addEventListener('click', reset);
stepButton.addEventListener('click', step);
playButton.addEventListener('click', play);
pauseButton.addEventListener('click', pause);

reset();
