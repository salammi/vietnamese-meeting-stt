const TARGET_RATE = 16000;
const FRAME_SAMPLES = 320; // 20 ms at 16 kHz
const statusBox = document.querySelector('.status');
const statusText = document.querySelector('#status');
const toggle = document.querySelector('#toggle');
const toggleText = document.querySelector('#toggleText');
const transcript = document.querySelector('#transcript');
const processing = document.querySelector('#processing');
const meter = document.querySelector('#meter');
const levelLabel = document.querySelector('#levelLabel');

let socket;
let stream;
let context;
let source;
let processor;
let running = false;
let pcmQueue = [];

class StreamingResampler {
  constructor(inputRate, outputRate) {
    this.ratio = inputRate / outputRate;
    this.carry = new Float32Array(0);
    this.position = 0;
  }
  process(input) {
    const samples = new Float32Array(this.carry.length + input.length);
    samples.set(this.carry);
    samples.set(input, this.carry.length);
    const result = [];
    while (this.position + 1 < samples.length) {
      const left = Math.floor(this.position);
      const fraction = this.position - left;
      result.push(samples[left] * (1 - fraction) + samples[left + 1] * fraction);
      this.position += this.ratio;
    }
    const consumed = Math.floor(this.position);
    this.carry = samples.slice(consumed);
    this.position -= consumed;
    return result;
  }
}

function setStatus(kind, text) {
  statusBox.className = `status ${kind}`;
  statusText.textContent = text;
}

function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  socket = new WebSocket(`${protocol}//${location.host}/ws/transcribe`);
  socket.binaryType = 'arraybuffer';
  socket.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (message.type === 'ready') {
      setStatus('ready', 'Sẵn sàng / Ready');
      toggle.disabled = false;
    } else if (message.type === 'processing') {
      processing.textContent = 'Đang nhận dạng…';
    } else if (message.type === 'transcript') {
      processing.textContent = '';
      appendSegment(message.segment);
    }
  };
  socket.onclose = () => {
    setStatus('error', 'Mất kết nối · thử lại…');
    toggle.disabled = true;
    setTimeout(connect, 1800);
  };
  socket.onerror = () => setStatus('error', 'Không thể kết nối');
}

function sendFrames(samples) {
  for (const sample of samples) pcmQueue.push(sample);
  while (pcmQueue.length >= FRAME_SAMPLES) {
    const frame = new Int16Array(FRAME_SAMPLES);
    for (let i = 0; i < FRAME_SAMPLES; i += 1) {
      const sample = Math.max(-1, Math.min(1, pcmQueue.shift()));
      frame[i] = sample < 0 ? sample * 32768 : sample * 32767;
    }
    if (socket.readyState === WebSocket.OPEN) socket.send(frame.buffer);
  }
}

async function startMicrophone() {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
  });
  context = new AudioContext();
  const resampler = new StreamingResampler(context.sampleRate, TARGET_RATE);
  source = context.createMediaStreamSource(stream);
  processor = context.createScriptProcessor(2048, 1, 1);
  processor.onaudioprocess = (event) => {
    const channel = event.inputBuffer.getChannelData(0);
    let energy = 0;
    for (const sample of channel) energy += sample * sample;
    const rms = Math.sqrt(energy / channel.length);
    meter.style.width = `${Math.min(100, rms * 650)}%`;
    sendFrames(resampler.process(channel));
  };
  source.connect(processor);
  processor.connect(context.destination);
  running = true;
  toggle.classList.add('active');
  toggleText.textContent = 'Tắt micro';
  levelLabel.textContent = 'Đang nghe · Listening';
}

async function stopMicrophone() {
  running = false;
  if (socket.readyState === WebSocket.OPEN) socket.send('flush');
  if (processor) processor.disconnect();
  if (source) source.disconnect();
  if (stream) stream.getTracks().forEach((track) => track.stop());
  if (context) await context.close();
  pcmQueue = [];
  meter.style.width = '0';
  toggle.classList.remove('active');
  toggleText.textContent = 'Bật micro';
  levelLabel.textContent = 'Đã tắt';
}

function formatTime(milliseconds) {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}

function appendSegment(segment) {
  transcript.querySelector('.empty')?.remove();
  const block = document.createElement('article');
  block.className = 'segment';
  const meta = document.createElement('div');
  meta.className = 'segment-meta';
  const fields = [
    formatTime(segment.started_at_ms),
    segment.language.toUpperCase(),
    `${Math.round(segment.confidence * 100)}%`,
  ];
  fields.forEach((value, index) => {
    const span = document.createElement('span');
    span.textContent = value;
    if (index > 0) span.className = 'tag';
    meta.appendChild(span);
  });
  if (segment.speaker_profile) {
    const profile = document.createElement('span');
    profile.className = 'tag';
    profile.title = 'Experimental acoustic classification; may be inaccurate.';
    profile.textContent = `${segment.speaker_profile.label} · ${Math.round(segment.speaker_profile.confidence * 100)}%`;
    meta.appendChild(profile);
  }
  const text = document.createElement('p');
  text.textContent = segment.text || '[Không nhận dạng được lời nói]';
  block.append(meta, text);
  transcript.appendChild(block);
  transcript.scrollTop = transcript.scrollHeight;
}

toggle.addEventListener('click', async () => {
  toggle.disabled = true;
  try {
    if (running) await stopMicrophone(); else await startMicrophone();
  } catch (error) {
    setStatus('error', `Lỗi micro: ${error.message}`);
  } finally {
    toggle.disabled = false;
  }
});

document.querySelector('#clear').addEventListener('click', () => {
  transcript.innerHTML = '<div class="empty"><span>VI ↔ EN</span><p>Bật micro và bắt đầu nói.<br>Turn on the microphone and start speaking.</p></div>';
});

connect();
