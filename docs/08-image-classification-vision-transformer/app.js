/* ==========================================================
   PROJECT 08 — TRAINED VISION TRANSFORMER BROWSER APP
   ========================================================== */

"use strict";


const ORT_VERSION = "1.27.0";

const ORT_DISTRIBUTION_URL =
    `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;


const state = {
    appData: null,
    preprocessor: null,
    labels: [],
    session: null,
    activeProvider: null,
    selectedFile: null,
    selectedImageUrl: null,
    selectedImageElement: null,
    currentAttentionFilter: "all"
};


const elements = {};


document.addEventListener(
    "DOMContentLoaded",
    initializeApplication
);


async function initializeApplication() {
    cacheElements();
    configureOnnxRuntime();
    bindUserInterfaceEvents();

    try {
        const [
            appData,
            preprocessor,
            labels
        ] = await Promise.all([
            fetchJson(
                "./assets/data/app_data.json"
            ),
            fetchJson(
                "./assets/models/preprocessor_config.json"
            ),
            fetchJson(
                "./assets/models/labels.json"
            )
        ]);

        state.appData = appData;
        state.preprocessor = preprocessor;
        state.labels = labels;

        validateApplicationConfiguration();

        renderProjectData();

        updateRuntimeStatus(
            "neutral",
            "Model is ready to load",
            `The validated ${formatNumber(
                state.appData.browser_model.model_size_mb,
                2
            )} MB ONNX model loads only when required.`
        );

        elements.loadModelButton.disabled = false;
    } catch (error) {
        console.error(error);

        updateRuntimeStatus(
            "error",
            "Application data could not be loaded",
            getErrorMessage(error)
        );
    }
}


function cacheElements() {
    const elementIds = [
        "hero-vit-accuracy",
        "hero-vit-f1",
        "hero-model-size",
        "hero-parity",
        "image-input",
        "drop-zone",
        "preview-container",
        "image-preview",
        "image-name",
        "image-dimensions",
        "runtime-status-badge",
        "runtime-status",
        "runtime-status-title",
        "runtime-status-message",
        "runtime-precision",
        "active-provider",
        "load-model-button",
        "predict-button",
        "prediction-results",
        "top-prediction-class",
        "top-prediction-confidence",
        "inference-time",
        "top-predictions-list",
        "vit-test-accuracy",
        "vit-test-f1",
        "vit-correct-count",
        "resnet-test-accuracy",
        "resnet-correct-count",
        "accuracy-difference",
        "comparison-table-body",
        "chart-gallery",
        "attention-notice",
        "attention-layers",
        "attention-heads",
        "attention-patch-grid",
        "attention-count",
        "correct-attention-grid",
        "incorrect-attention-grid",
        "attention-gallery",
        "limitations-list",
        "preprocessing-canvas"
    ];

    for (const elementId of elementIds) {
        elements[
            camelCase(elementId)
        ] = document.getElementById(
            elementId
        );
    }

    elements.filterButtons = Array.from(
        document.querySelectorAll(
            ".filter-button"
        )
    );
}


function configureOnnxRuntime() {
    if (typeof ort === "undefined") {
        throw new Error(
            "ONNX Runtime Web did not load."
        );
    }

    ort.env.logLevel = "warning";

    /*
     * GitHub Pages does not provide cross-origin isolation
     * headers for SharedArrayBuffer-based multithreading.
     * Single-threaded WASM is therefore the safest fallback.
     */
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = ORT_DISTRIBUTION_URL;
}


function bindUserInterfaceEvents() {
    elements.dropZone.addEventListener(
        "click",
        () => elements.imageInput.click()
    );

    elements.dropZone.addEventListener(
        "keydown",
        event => {
            if (
                event.key === "Enter"
                || event.key === " "
            ) {
                event.preventDefault();
                elements.imageInput.click();
            }
        }
    );

    elements.imageInput.addEventListener(
        "change",
        event => {
            const selectedFile =
                event.target.files?.[0];

            if (selectedFile) {
                handleSelectedImage(
                    selectedFile
                );
            }
        }
    );

    for (
        const eventName
        of ["dragenter", "dragover"]
    ) {
        elements.dropZone.addEventListener(
            eventName,
            event => {
                event.preventDefault();
                elements.dropZone.classList.add(
                    "dragging"
                );
            }
        );
    }

    for (
        const eventName
        of ["dragleave", "drop"]
    ) {
        elements.dropZone.addEventListener(
            eventName,
            event => {
                event.preventDefault();
                elements.dropZone.classList.remove(
                    "dragging"
                );
            }
        );
    }

    elements.dropZone.addEventListener(
        "drop",
        event => {
            const droppedFile =
                event.dataTransfer?.files?.[0];

            if (droppedFile) {
                handleSelectedImage(
                    droppedFile
                );
            }
        }
    );

    elements.loadModelButton.addEventListener(
        "click",
        initializeModelSession
    );

    elements.predictButton.addEventListener(
        "click",
        runPrediction
    );

    for (
        const filterButton
        of elements.filterButtons
    ) {
        filterButton.addEventListener(
            "click",
            () => {
                state.currentAttentionFilter =
                    filterButton.dataset.filter;

                for (
                    const button
                    of elements.filterButtons
                ) {
                    button.classList.toggle(
                        "active",
                        button === filterButton
                    );
                }

                renderAttentionGallery();
            }
        );
    }
}


function validateApplicationConfiguration() {
    if (
        !Array.isArray(state.labels)
        || state.labels.length !== 10
    ) {
        throw new Error(
            "The deployment label file must contain ten CIFAR-10 classes."
        );
    }

    const browserModel =
        state.appData?.browser_model;

    if (!browserModel?.model_url) {
        throw new Error(
            "app_data.json does not contain the browser model URL."
        );
    }

    if (
        browserModel.input_name
        !== "pixel_values"
    ) {
        throw new Error(
            "Unexpected ONNX input name."
        );
    }

    if (
        browserModel.output_name
        !== "logits"
    ) {
        throw new Error(
            "Unexpected ONNX output name."
        );
    }
}


async function initializeModelSession() {
    if (state.session) {
        return state.session;
    }

    elements.loadModelButton.disabled = true;
    elements.predictButton.disabled = true;

    updateRuntimeStatus(
        "loading",
        "Loading trained ONNX model",
        "Downloading and initializing the validated CIFAR-10 model…"
    );

    const modelUrl = new URL(
        state.appData.browser_model.model_url,
        document.baseURI
    ).href;

    let webGpuError = null;

    if ("gpu" in navigator) {
        try {
            updateRuntimeStatus(
                "loading",
                "Initializing WebGPU",
                "Attempting hardware-accelerated browser inference…"
            );

            state.session =
                await ort.InferenceSession.create(
                    modelUrl,
                    {
                        executionProviders: [
                            "webgpu"
                        ],
                        graphOptimizationLevel:
                            "all"
                    }
                );

            state.activeProvider = "WebGPU";
        } catch (error) {
            webGpuError = error;

            console.warn(
                "WebGPU initialization failed. Falling back to WASM.",
                error
            );
        }
    }

    if (!state.session) {
        updateRuntimeStatus(
            "loading",
            "Initializing WebAssembly",
            "Using the portable CPU fallback runtime…"
        );

        try {
            state.session =
                await ort.InferenceSession.create(
                    modelUrl,
                    {
                        executionProviders: [
                            "wasm"
                        ],
                        graphOptimizationLevel:
                            "all"
                    }
                );

            state.activeProvider = "WASM";
        } catch (wasmError) {
            elements.loadModelButton.disabled = false;

            const combinedMessage = webGpuError
                ? `WebGPU: ${getErrorMessage(
                    webGpuError
                )}. WASM: ${getErrorMessage(
                    wasmError
                )}`
                : getErrorMessage(wasmError);

            updateRuntimeStatus(
                "error",
                "Model initialization failed",
                combinedMessage
            );

            throw wasmError;
        }
    }

    validateSessionMetadata();

    elements.activeProvider.textContent =
        state.activeProvider;

    elements.loadModelButton.textContent =
        "Model loaded";

    elements.loadModelButton.disabled = true;

    elements.predictButton.disabled =
        !state.selectedFile;

    updateRuntimeStatus(
        "ready",
        `${state.activeProvider} model ready`,
        "The trained CIFAR-10 model is loaded and ready for local inference."
    );

    return state.session;
}


function validateSessionMetadata() {
    const expectedInput =
        state.appData.browser_model.input_name;

    const expectedOutput =
        state.appData.browser_model.output_name;

    if (
        !state.session.inputNames.includes(
            expectedInput
        )
    ) {
        throw new Error(
            `Model input '${expectedInput}' was not found.`
        );
    }

    if (
        !state.session.outputNames.includes(
            expectedOutput
        )
    ) {
        throw new Error(
            `Model output '${expectedOutput}' was not found.`
        );
    }
}


async function handleSelectedImage(file) {
    const allowedTypes = new Set([
        "image/png",
        "image/jpeg",
        "image/webp"
    ]);

    if (!allowedTypes.has(file.type)) {
        updateRuntimeStatus(
            "error",
            "Unsupported image type",
            "Please select a PNG, JPEG or WebP image."
        );

        return;
    }

    if (
        state.selectedImageUrl
    ) {
        URL.revokeObjectURL(
            state.selectedImageUrl
        );
    }

    state.selectedFile = file;
    state.selectedImageUrl =
        URL.createObjectURL(file);

    elements.imagePreview.src =
        state.selectedImageUrl;

    await waitForImage(
        elements.imagePreview
    );

    state.selectedImageElement =
        elements.imagePreview;

    elements.imageName.textContent =
        file.name;

    elements.imageDimensions.textContent =
        `${elements.imagePreview.naturalWidth} × `
        + `${elements.imagePreview.naturalHeight}`;

    elements.dropZone.classList.add(
        "hidden"
    );

    elements.previewContainer.classList.remove(
        "hidden"
    );

    elements.predictionResults.classList.add(
        "hidden"
    );

    elements.predictButton.disabled =
        !state.session;

    if (!state.session) {
        updateRuntimeStatus(
            "neutral",
            "Image selected",
            "Load the trained model before running classification."
        );
    }
}


async function runPrediction() {
    if (!state.selectedImageElement) {
        updateRuntimeStatus(
            "error",
            "No image selected",
            "Select an image before running inference."
        );

        return;
    }

    try {
        if (!state.session) {
            await initializeModelSession();
        }

        elements.predictButton.disabled = true;
        elements.predictButton.textContent =
            "Classifying…";

        updateRuntimeStatus(
            "loading",
            "Running local inference",
            "Preprocessing the image and executing the Vision Transformer…"
        );

        const inputTensor =
            preprocessImage(
                state.selectedImageElement
            );

        const inputName =
            state.appData.browser_model.input_name;

        const outputName =
            state.appData.browser_model.output_name;

        const inferenceStarted =
            performance.now();

        const outputs =
            await state.session.run({
                [inputName]: inputTensor
            });

        const elapsedMilliseconds =
            performance.now()
            - inferenceStarted;

        const outputTensor =
            outputs[outputName];

        if (!outputTensor) {
            throw new Error(
                `Inference output '${outputName}' was not returned.`
            );
        }

        const outputData =
            typeof outputTensor.getData
                === "function"
                ? await outputTensor.getData()
                : outputTensor.data;

        const logits = Array.from(
            outputData
        );

        if (
            logits.length
            !== state.labels.length
        ) {
            throw new Error(
                `Expected ${state.labels.length} logits but received ${logits.length}.`
            );
        }

        const probabilities =
            softmax(logits);

        const rankedPredictions =
            probabilities
            .map(
                (probability, classId) => ({
                    classId,
                    className:
                        state.labels[classId],
                    probability
                })
            )
            .sort(
                (first, second) =>
                    second.probability
                    - first.probability
            );

        renderPredictionResults(
            rankedPredictions,
            elapsedMilliseconds
        );

        updateRuntimeStatus(
            "ready",
            "Prediction completed",
            `Inference ran locally with ${state.activeProvider}.`
        );
    } catch (error) {
        console.error(error);

        updateRuntimeStatus(
            "error",
            "Prediction failed",
            getErrorMessage(error)
        );
    } finally {
        elements.predictButton.disabled =
            !state.selectedFile
            || !state.session;

        elements.predictButton.textContent =
            "Classify image";
    }
}


function preprocessImage(imageElement) {
    const canvas =
        elements.preprocessingCanvas;

    const context = canvas.getContext(
        "2d",
        {
            willReadFrequently: true
        }
    );

    const imageSize = 224;

    canvas.width = imageSize;
    canvas.height = imageSize;

    context.clearRect(
        0,
        0,
        imageSize,
        imageSize
    );

    /*
     * The Python evaluation transform directly resized images
     * to 224 × 224. The browser repeats that same operation.
     */
    context.drawImage(
        imageElement,
        0,
        0,
        imageSize,
        imageSize
    );

    const rgbaPixels =
        context.getImageData(
            0,
            0,
            imageSize,
            imageSize
        ).data;

    const mean =
        state.preprocessor.image_mean
        ?? [0.5, 0.5, 0.5];

    const standardDeviation =
        state.preprocessor.image_std
        ?? [0.5, 0.5, 0.5];

    const rescaleFactor =
        state.preprocessor.rescale_factor
        ?? (1 / 255);

    const channelSize =
        imageSize * imageSize;

    const tensorData =
        new Float32Array(
            3 * channelSize
        );

    for (
        let pixelIndex = 0;
        pixelIndex < channelSize;
        pixelIndex += 1
    ) {
        const rgbaIndex =
            pixelIndex * 4;

        const red =
            rgbaPixels[rgbaIndex]
            * rescaleFactor;

        const green =
            rgbaPixels[rgbaIndex + 1]
            * rescaleFactor;

        const blue =
            rgbaPixels[rgbaIndex + 2]
            * rescaleFactor;

        tensorData[pixelIndex] =
            (
                red
                - mean[0]
            )
            / standardDeviation[0];

        tensorData[
            channelSize + pixelIndex
        ] =
            (
                green
                - mean[1]
            )
            / standardDeviation[1];

        tensorData[
            (2 * channelSize)
            + pixelIndex
        ] =
            (
                blue
                - mean[2]
            )
            / standardDeviation[2];
    }

    return new ort.Tensor(
        "float32",
        tensorData,
        [
            1,
            3,
            imageSize,
            imageSize
        ]
    );
}


function softmax(logits) {
    const maximumLogit =
        Math.max(...logits);

    const exponentials =
        logits.map(
            logit =>
                Math.exp(
                    logit
                    - maximumLogit
                )
        );

    const exponentialSum =
        exponentials.reduce(
            (runningTotal, value) =>
                runningTotal + value,
            0
        );

    return exponentials.map(
        value =>
            value / exponentialSum
    );
}


function renderPredictionResults(
    rankedPredictions,
    elapsedMilliseconds
) {
    const topPrediction =
        rankedPredictions[0];

    elements.topPredictionClass.textContent =
        topPrediction.className;

    elements.topPredictionConfidence.textContent =
        `${formatPercent(
            topPrediction.probability,
            2
        )} confidence`;

    elements.inferenceTime.textContent =
        `${elapsedMilliseconds.toFixed(2)} ms`;

    elements.topPredictionsList.replaceChildren();

    for (
        const prediction
        of rankedPredictions.slice(0, 3)
    ) {
        const row =
            document.createElement("div");

        row.className =
            "prediction-row";

        const label =
            document.createElement("span");

        label.className =
            "prediction-label";

        label.textContent =
            prediction.className;

        const track =
            document.createElement("div");

        track.className =
            "prediction-track";

        const fill =
            document.createElement("div");

        fill.className =
            "prediction-fill";

        fill.style.width =
            `${Math.max(
                prediction.probability * 100,
                0.5
            )}%`;

        track.appendChild(fill);

        const percentage =
            document.createElement("span");

        percentage.className =
            "prediction-percentage";

        percentage.textContent =
            formatPercent(
                prediction.probability,
                2
            );

        row.append(
            label,
            track,
            percentage
        );

        elements.topPredictionsList.appendChild(
            row
        );
    }

    elements.predictionResults.classList.remove(
        "hidden"
    );
}


function renderProjectData() {
    const testResults =
        state.appData.test_results;

    const vit =
        testResults.vision_transformer;

    const resnet =
        testResults.resnet18;

    const parity =
        state.appData.onnx_parity
        ?.observed_results;

    setText(
        elements.heroVitAccuracy,
        formatPercent(vit.accuracy, 2)
    );

    setText(
        elements.heroVitF1,
        formatNumber(vit.macro_f1, 4)
    );

    setText(
        elements.heroModelSize,
        `${formatNumber(
            state.appData.browser_model.model_size_mb,
            2
        )} MB`
    );

    setText(
        elements.heroParity,
        formatPercent(
            parity?.prediction_agreement,
            2
        )
    );

    setText(
        elements.runtimePrecision,
        String(
            state.appData.browser_model.precision
        ).toUpperCase()
    );

    setText(
        elements.vitTestAccuracy,
        formatPercent(vit.accuracy, 2)
    );

    setText(
        elements.vitTestF1,
        formatNumber(vit.macro_f1, 4)
    );

    setText(
        elements.vitCorrectCount,
        `${formatInteger(
            vit.correct_predictions
        )} of 10,000 correct`
    );

    setText(
        elements.resnetTestAccuracy,
        formatPercent(resnet.accuracy, 2)
    );

    setText(
        elements.resnetCorrectCount,
        `${formatInteger(
            resnet.correct_predictions
        )} of 10,000 correct`
    );

    setText(
        elements.accuracyDifference,
        formatSignedPercentagePoints(
            testResults
                .accuracy_difference_vit_minus_resnet
        )
    );

    renderComparisonTable();
    renderChartGallery();
    renderAttentionSection();
    renderLimitations();
}


function renderComparisonTable() {
    elements.comparisonTableBody.replaceChildren();

    const comparisonRecords =
        Array.isArray(
            state.appData.model_comparison
        )
            ? state.appData.model_comparison
            : [];

    for (
        const record
        of comparisonRecords
    ) {
        const row =
            document.createElement("tr");

        const values = [
            record.model,
            formatPercent(
                record.test_accuracy,
                2
            ),
            formatNumber(
                record.macro_f1,
                4
            ),
            formatCompactInteger(
                record.parameters
            ),
            `${formatNumber(
                record.model_size_mb,
                2
            )} MB`,
            record.average_latency_ms == null
                ? "—"
                : `${formatNumber(
                    record.average_latency_ms,
                    3
                )} ms`
        ];

        for (const value of values) {
            const cell =
                document.createElement("td");

            cell.textContent =
                value ?? "—";

            row.appendChild(cell);
        }

        elements.comparisonTableBody.appendChild(
            row
        );
    }

    if (
        comparisonRecords.length === 0
    ) {
        const row =
            document.createElement("tr");

        const cell =
            document.createElement("td");

        cell.colSpan = 6;
        cell.textContent =
            "Comparison data is unavailable.";

        row.appendChild(cell);

        elements.comparisonTableBody.appendChild(
            row
        );
    }
}


function renderChartGallery() {
    elements.chartGallery.replaceChildren();

    const preferredCharts = [
        [
            "model_comparison",
            "Test performance comparison"
        ],
        [
            "latency_comparison",
            "Local RTX 5090 latency comparison"
        ],
        [
            "parameter_comparison",
            "Parameter-count comparison"
        ],
        [
            "vit_normalized_confusion_matrix",
            "ViT normalized confusion matrix"
        ],
        [
            "resnet_normalized_confusion_matrix",
            "ResNet-18 normalized confusion matrix"
        ],
        [
            "quantization_size_comparison",
            "ONNX model-size analysis"
        ]
    ];

    for (
        const [
            chartKey,
            chartTitle
        ]
        of preferredCharts
    ) {
        const imageUrl =
            state.appData.chart_urls?.[
                chartKey
            ];

        if (!imageUrl) {
            continue;
        }

        const article =
            document.createElement("article");

        article.className =
            "chart-card";

        const image =
            document.createElement("img");

        image.src = imageUrl;
        image.alt = chartTitle;
        image.loading = "lazy";

        const content =
            document.createElement("div");

        const title =
            document.createElement("strong");

        title.textContent = chartTitle;

        const description =
            document.createElement("span");

        description.textContent =
            "Generated from the real training and evaluation pipeline.";

        content.append(
            title,
            description
        );

        article.append(
            image,
            content
        );

        elements.chartGallery.appendChild(
            article
        );
    }
}


function renderAttentionSection() {
    const attention =
        state.appData.attention_rollout;

    setText(
        elements.attentionLayers,
        attention.transformer_layers
    );

    setText(
        elements.attentionHeads,
        attention.attention_heads
    );

    setText(
        elements.attentionPatchGrid,
        Array.isArray(
            attention.patch_grid_size
        )
            ? attention.patch_grid_size.join(
                " × "
            )
            : "—"
    );

    const totalExamples =
        Number(
            attention.correct_examples_generated
            ?? 0
        )
        +
        Number(
            attention.incorrect_examples_generated
            ?? 0
        );

    setText(
        elements.attentionCount,
        totalExamples
    );

    elements.attentionNotice.textContent =
        attention.interpretation_notice;

    elements.correctAttentionGrid.src =
        attention.correct_grid_url;

    elements.incorrectAttentionGrid.src =
        attention.incorrect_grid_url;

    renderAttentionGallery();
}


function renderAttentionGallery() {
    if (!state.appData) {
        return;
    }

    const attentionExamples =
        state.appData
            .attention_rollout
            .examples
        ?? [];

    const visibleExamples =
        state.currentAttentionFilter
        === "all"
            ? attentionExamples
            : attentionExamples.filter(
                example =>
                    example.selection_type
                    === state.currentAttentionFilter
            );

    elements.attentionGallery.replaceChildren();

    for (
        const example
        of visibleExamples
    ) {
        const article =
            document.createElement("article");

        article.className =
            "attention-card";

        const image =
            document.createElement("img");

        image.src =
            example.image_url;

        image.alt =
            `Attention rollout: actual ${example.actual_class}, predicted ${example.predicted_class}`;

        image.loading = "lazy";

        const content =
            document.createElement("div");

        content.className =
            "attention-card-content";

        const status =
            document.createElement("span");

        status.className =
            `result-pill ${example.selection_type}`;

        status.textContent =
            example.selection_type;

        const prediction =
            document.createElement("strong");

        prediction.textContent =
            `${example.actual_class} → ${example.predicted_class}`;

        const confidence =
            document.createElement("span");

        confidence.textContent =
            `${formatPercent(
                example.confidence,
                2
            )} confidence`;

        content.append(
            status,
            prediction,
            confidence
        );

        article.append(
            image,
            content
        );

        elements.attentionGallery.appendChild(
            article
        );
    }
}


function renderLimitations() {
    elements.limitationsList.replaceChildren();

    for (
        const limitation
        of state.appData.limitations ?? []
    ) {
        const item =
            document.createElement("li");

        item.textContent =
            limitation;

        elements.limitationsList.appendChild(
            item
        );
    }
}


function updateRuntimeStatus(
    statusType,
    title,
    message
) {
    elements.runtimeStatus.className =
        `runtime-status ${statusType}`;

    elements.runtimeStatusBadge.className =
        `status-badge ${statusType}`;

    elements.runtimeStatusTitle.textContent =
        title;

    elements.runtimeStatusMessage.textContent =
        message;

    const badgeLabels = {
        neutral: "Not loaded",
        loading: "Working",
        ready: "Ready",
        error: "Error"
    };

    elements.runtimeStatusBadge.textContent =
        badgeLabels[statusType]
        ?? statusType;
}


async function fetchJson(url) {
    const response = await fetch(
        url,
        {
            cache: "no-cache"
        }
    );

    if (!response.ok) {
        throw new Error(
            `Could not load ${url}: HTTP ${response.status}`
        );
    }

    return response.json();
}


function waitForImage(imageElement) {
    if (
        imageElement.complete
        && imageElement.naturalWidth > 0
    ) {
        return Promise.resolve();
    }

    return new Promise(
        (resolve, reject) => {
            imageElement.addEventListener(
                "load",
                resolve,
                {
                    once: true
                }
            );

            imageElement.addEventListener(
                "error",
                () => reject(
                    new Error(
                        "The selected image could not be decoded."
                    )
                ),
                {
                    once: true
                }
            );
        }
    );
}


function setText(
    element,
    value
) {
    element.textContent =
        value == null
            ? "—"
            : String(value);
}


function formatPercent(
    value,
    decimalPlaces = 2
) {
    const numericValue =
        Number(value);

    if (
        !Number.isFinite(
            numericValue
        )
    ) {
        return "—";
    }

    return `${(
        numericValue * 100
    ).toFixed(decimalPlaces)}%`;
}


function formatSignedPercentagePoints(
    value
) {
    const numericValue =
        Number(value);

    if (
        !Number.isFinite(
            numericValue
        )
    ) {
        return "—";
    }

    const percentagePoints =
        numericValue * 100;

    const sign =
        percentagePoints >= 0
            ? "+"
            : "";

    return `${sign}${percentagePoints.toFixed(
        2
    )} pp`;
}


function formatNumber(
    value,
    decimalPlaces = 2
) {
    const numericValue =
        Number(value);

    if (
        !Number.isFinite(
            numericValue
        )
    ) {
        return "—";
    }

    return numericValue.toFixed(
        decimalPlaces
    );
}


function formatInteger(value) {
    const numericValue =
        Number(value);

    if (
        !Number.isFinite(
            numericValue
        )
    ) {
        return "—";
    }

    return Math.round(
        numericValue
    ).toLocaleString("en-US");
}


function formatCompactInteger(value) {
    const numericValue =
        Number(value);

    if (
        !Number.isFinite(
            numericValue
        )
    ) {
        return "—";
    }

    return new Intl.NumberFormat(
        "en-US",
        {
            notation: "compact",
            maximumFractionDigits: 2
        }
    ).format(numericValue);
}


function camelCase(hyphenatedText) {
    return hyphenatedText.replace(
        /-([a-z])/g,
        (
            _fullMatch,
            capturedLetter
        ) =>
            capturedLetter.toUpperCase()
    );
}


function getErrorMessage(error) {
    if (
        error instanceof Error
    ) {
        return error.message;
    }

    return String(error);
}
