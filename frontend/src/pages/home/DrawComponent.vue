<template>
<div class="paint-app">
	<canvas
	ref="canvas"
	class="canvas"
	:width="CANVAS_SIZE"
	:height="CANVAS_SIZE"
	@mousemove="onMouseMove"
	@mousedown="startPainting"
	@mouseup="stopPainting"
	@mouseleave="stopPainting"
	@touchstart.prevent="handleTouchStart"
	@touchmove.prevent="handleTouchMove"
	@touchend.prevent="stopPainting"
	@touchcancel.prevent="stopPainting"
	@click="handleCanvasClick"
	@contextmenu.prevent="handleCM"
	></canvas>

	<div class="controls mb-6">
	<div class="controls_btns">
		<button @click="handleTryAgainClick">Thử lại</button>
		<button @click="handleSaveClick">Dự đoán ngay</button>
	</div>

	<div class="controls_colors">
		<div
		v-for="color in colors"
		:key="color"
		class="controls_color"
		:style="{ backgroundColor: color }"
		@click="handleColorClick(color)"
		></div>
	</div>
	</div>
</div>
</template>

<script setup>
import { ref, onMounted, defineEmits } from 'vue';

const emit = defineEmits(['update'])

// Constants
const INITIAL_COLOR = 'white';
const CANVAS_SIZE = 350;

// Colors array
const colors = [];

// Refs
const canvas = ref(null);
const ctx = ref(null);
const painting = ref(false);
const filling = ref(false);

// Methods for mouse events
const stopPainting = () => {
painting.value = false;
};

const startPainting = () => {
painting.value = true;
};

// Convert touch coordinates to canvas coordinates
const getTouchPos = (canvasEl, touch) => {
const rect = canvasEl.getBoundingClientRect();
return {
	x: touch.clientX - rect.left,
	y: touch.clientY - rect.top
};
};

// Touch event handlers
const handleTouchStart = (event) => {
if (!ctx.value || !canvas.value) return;

const touch = event.touches[0];
const { x, y } = getTouchPos(canvas.value, touch);

painting.value = true;
ctx.value.beginPath();
ctx.value.moveTo(x, y);
};

const handleTouchMove = (event) => {
if (!ctx.value || !canvas.value || !painting.value) return;

const touch = event.touches[0];
const { x, y } = getTouchPos(canvas.value, touch);

ctx.value.lineTo(x, y);
ctx.value.stroke();
};

// Mouse event handler
const onMouseMove = (event) => {
if (!ctx.value) return;

const x = event.offsetX;
const y = event.offsetY;

if (!painting.value) {
	ctx.value.beginPath();
	ctx.value.moveTo(x, y);
} else {
	ctx.value.lineTo(x, y);
	ctx.value.stroke();
}
};

const handleTryAgainClick = () => {
if (!ctx.value || !canvas.value) return;

ctx.value.fillStyle = 'black';
ctx.value.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

// Reset drawing style
ctx.value.strokeStyle = INITIAL_COLOR;
ctx.value.lineWidth = 25;
ctx.value.lineCap = "round";
ctx.value.shadowColor = INITIAL_COLOR;
ctx.value.shadowBlur = 3;
};

const handleCanvasClick = () => {
if (!ctx.value) return;
if (filling.value) {
	ctx.value.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
}
};

const handleCM = (event) => {
event.preventDefault();
};

const handleSaveClick = () => {
if (!canvas.value) return;
const base64Data = canvas.value.toDataURL('image/png').split(',')[1];
const binaryData = new Uint8Array(atob(base64Data).split('').map(char => char.charCodeAt(0)));
const blob = new Blob([binaryData], { type: 'image/png' });
emit('update', blob)
};

// Initialize canvas on mount
onMounted(() => {
if (!canvas.value) return;

ctx.value = canvas.value.getContext('2d');

// Set initial canvas background
ctx.value.fillStyle = 'black';
ctx.value.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

// Set initial drawing styles
ctx.value.strokeStyle = INITIAL_COLOR;
ctx.value.fillStyle = INITIAL_COLOR;
ctx.value.lineWidth = 25;
ctx.value.lineCap = "round";
ctx.value.shadowColor = INITIAL_COLOR;
ctx.value.shadowBlur = 3;
ctx.value.shadowOffsetX = 0;
ctx.value.shadowOffsetY = 0;
});
</script>

<style scoped>
.paint-app {
	display: flex;
	flex-direction: column;
	align-items: center;
	padding: 1rem;
	padding-top: 0rem;
}

.canvas {
	width: 350px;
	background-color: black;
	border-radius: 15px;
	box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
}

.controls {
	margin-top: 1rem;
	display: flex;
	flex-direction: column;
	align-items: center;
	gap: 1rem;
}

.controls_range {
	width: 200px;
}

.controls_range input {
	width: 100%;
}

.controls_btns {
	display: flex;
	gap: 0.5rem;
}

.controls_btns button {
	all: unset;
	cursor: pointer;
	background-color: white;
	padding: 0.5rem 1rem;
	text-transform: uppercase;
	font-weight: 600;
	font-size: 0.875rem;
	border-radius: 0.375rem;
	box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
	border: 2px solid rgba(0, 0, 0, 0.2);
	color: rgba(0, 0, 0, 0.7);
	transition: all 0.2s ease-in-out;
}

.controls_btns button:hover {
	background-color: rgba(0, 0, 0, 0.05);
}

.controls_btns button:active {
	transform: scale(0.98);
}

.controls_colors {
	display: flex;
	gap: 0.5rem;
}

.controls_color {
	width: 2.5rem;
	height: 2.5rem;
	border-radius: 1.25rem;
	cursor: pointer;
	box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
	transition: transform 0.2s ease-in-out;
}

.controls_color:hover {
	transform: scale(1.1);
}

.controls_color:active {
	transform: scale(0.95);
}
</style>