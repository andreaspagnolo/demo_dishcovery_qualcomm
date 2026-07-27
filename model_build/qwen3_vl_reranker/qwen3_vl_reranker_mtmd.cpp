#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {

constexpr const char * DEFAULT_SYSTEM_PROMPT =
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    "Note that the answer can only be \"yes\" or \"no\".";

constexpr const char * DEFAULT_INSTRUCTION =
    "Carefully compare the food image with the candidate caption. Score high only when the visible dish, "
    "ingredients, cooking method, sauce, plating, and other visual details match the caption.";

struct options {
    std::string model_path;
    std::string mmproj_path;
    std::string image_path;
    std::string query_text;
    std::string document_text;
    std::string instruction = DEFAULT_INSTRUCTION;
    std::string system_prompt = DEFAULT_SYSTEM_PROMPT;
    std::string backend_dir = "/home/ubuntu/.local/share/geniex/llama_cpp";
    std::string compute = "npu";
    std::string flash_attention = "off";
    int32_t n_ctx = 4096;
    int32_t n_batch = 512;
    int32_t n_threads = 8;
    int32_t image_min_tokens = -1;
    int32_t image_max_tokens = 512;
    int32_t max_batch_candidates = 4;
    bool list_devices = false;
    bool interactive = false;
    bool print_timings = false;
    bool verbose = false;
};

struct model_deleter {
    void operator()(llama_model * ptr) const { if (ptr) llama_model_free(ptr); }
};

struct context_deleter {
    void operator()(llama_context * ptr) const { if (ptr) llama_free(ptr); }
};

struct mtmd_deleter {
    void operator()(mtmd_context * ptr) const { if (ptr) mtmd_free(ptr); }
};

struct bitmap_deleter {
    void operator()(mtmd_bitmap * ptr) const { if (ptr) mtmd_bitmap_free(ptr); }
};

struct chunks_deleter {
    void operator()(mtmd_input_chunks * ptr) const { if (ptr) mtmd_input_chunks_free(ptr); }
};

using model_ptr = std::unique_ptr<llama_model, model_deleter>;
using context_ptr = std::unique_ptr<llama_context, context_deleter>;
using mtmd_ptr = std::unique_ptr<mtmd_context, mtmd_deleter>;
using bitmap_ptr = std::unique_ptr<mtmd_bitmap, bitmap_deleter>;
using chunks_ptr = std::unique_ptr<mtmd_input_chunks, chunks_deleter>;

std::string require_value(int & i, int argc, char ** argv, const std::string & flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + flag);
    }
    return argv[++i];
}

int32_t parse_i32(const std::string & value, const std::string & flag) {
    size_t used = 0;
    long parsed = std::stol(value, &used);
    if (used != value.size() || parsed < std::numeric_limits<int32_t>::min() || parsed > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("Invalid integer for " + flag + ": " + value);
    }
    return static_cast<int32_t>(parsed);
}

void print_help(const char * argv0) {
    std::cout
        << "Usage: " << argv0 << " --model MODEL.gguf --mmproj MMPROJ.gguf --image IMAGE --document TEXT [options]\n\n"
        << "Runs Qwen3-VL-Reranker through llama.cpp libmtmd and returns\n"
        << "sigmoid(logit[yes] - logit[no]).\n\n"
        << "Options:\n"
        << "  --query TEXT                 Optional text alongside the query image\n"
        << "  --instruction TEXT           Retrieval instruction\n"
        << "  --system-prompt TEXT         Override the official reranker system prompt\n"
        << "  --compute npu|cpu|hybrid     Text-model placement (default: npu)\n"
        << "  --backend-dir DIR            GenieX llama.cpp shared-library directory\n"
        << "  --n-ctx N                    Context size (default: 4096)\n"
        << "  --n-batch N                  Decode batch size (default: 512)\n"
        << "  --threads N                  CPU worker threads (default: 8)\n"
        << "  --flash-attention off|on|auto Flash-attention policy (default: off)\n"
        << "  --image-min-tokens N         Dynamic image lower bound (-1: metadata default)\n"
        << "  --image-max-tokens N         Dynamic image upper bound (default: 512)\n"
        << "  --max-batch-candidates N     Candidate sequences scored together (default: 4; HTP max: 4)\n"
        << "  --interactive                Emit ready JSON, then read tab-separated rows from stdin\n"
        << "  --print-timings              Enable mtmd timing logs\n"
        << "  --verbose                    Show llama/ggml backend diagnostics\n"
        << "  --list-devices               Print registered backend devices and exit\n"
        << "  -h, --help                   Show this help\n";
}

options parse_args(int argc, char ** argv) {
    options out;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "-h" || flag == "--help") {
            print_help(argv[0]);
            std::exit(0);
        } else if (flag == "--model") {
            out.model_path = require_value(i, argc, argv, flag);
        } else if (flag == "--mmproj") {
            out.mmproj_path = require_value(i, argc, argv, flag);
        } else if (flag == "--image") {
            out.image_path = require_value(i, argc, argv, flag);
        } else if (flag == "--query") {
            out.query_text = require_value(i, argc, argv, flag);
        } else if (flag == "--document") {
            out.document_text = require_value(i, argc, argv, flag);
        } else if (flag == "--instruction") {
            out.instruction = require_value(i, argc, argv, flag);
        } else if (flag == "--system-prompt") {
            out.system_prompt = require_value(i, argc, argv, flag);
        } else if (flag == "--backend-dir") {
            out.backend_dir = require_value(i, argc, argv, flag);
        } else if (flag == "--compute") {
            out.compute = require_value(i, argc, argv, flag);
        } else if (flag == "--n-ctx") {
            out.n_ctx = parse_i32(require_value(i, argc, argv, flag), flag);
        } else if (flag == "--n-batch") {
            out.n_batch = parse_i32(require_value(i, argc, argv, flag), flag);
        } else if (flag == "--threads") {
            out.n_threads = parse_i32(require_value(i, argc, argv, flag), flag);
        } else if (flag == "--flash-attention") {
            out.flash_attention = require_value(i, argc, argv, flag);
        } else if (flag == "--image-min-tokens") {
            out.image_min_tokens = parse_i32(require_value(i, argc, argv, flag), flag);
        } else if (flag == "--image-max-tokens") {
            out.image_max_tokens = parse_i32(require_value(i, argc, argv, flag), flag);
        } else if (flag == "--max-batch-candidates") {
            out.max_batch_candidates = parse_i32(require_value(i, argc, argv, flag), flag);
        } else if (flag == "--list-devices") {
            out.list_devices = true;
        } else if (flag == "--interactive") {
            out.interactive = true;
        } else if (flag == "--print-timings") {
            out.print_timings = true;
        } else if (flag == "--verbose") {
            out.verbose = true;
        } else {
            throw std::runtime_error("Unknown argument: " + flag);
        }
    }
    if (out.compute != "npu" && out.compute != "cpu" && out.compute != "hybrid") {
        throw std::runtime_error("--compute must be npu, cpu, or hybrid");
    }
    if (out.flash_attention != "off" && out.flash_attention != "on" && out.flash_attention != "auto") {
        throw std::runtime_error("--flash-attention must be off, on, or auto");
    }
    if (out.n_ctx <= 0 || out.n_batch <= 0 || out.n_threads <= 0 || out.max_batch_candidates <= 0) {
        throw std::runtime_error("--n-ctx, --n-batch, --threads, and --max-batch-candidates must be positive");
    }
    if (out.compute != "cpu" && out.max_batch_candidates > 4) {
        throw std::runtime_error("GenieX/HTP supports at most 4 reliable shared-prefix candidate sequences");
    }
    return out;
}

void warning_log_callback(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

struct backend_guard {
    backend_guard() { llama_backend_init(); }
    ~backend_guard() { llama_backend_free(); }
    backend_guard(const backend_guard &) = delete;
    backend_guard & operator=(const backend_guard &) = delete;
};

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

const char * device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU: return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accelerator";
        default: return "unknown";
    }
}

bool is_npu_device(ggml_backend_dev_t dev) {
    const std::string name = lower(ggml_backend_dev_name(dev));
    const std::string description = lower(ggml_backend_dev_description(dev));
    return name.find("hexagon") != std::string::npos || name.find("htp") != std::string::npos ||
           description.find("hexagon") != std::string::npos || description.find("htp") != std::string::npos;
}

void print_devices() {
    const size_t count = ggml_backend_dev_count();
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            std::cout << i << "\t<unavailable>\tunknown\tbackend initialization failed\n";
            continue;
        }
        std::cout << i << '\t' << ggml_backend_dev_name(dev) << '\t'
                  << device_type_name(ggml_backend_dev_type(dev)) << '\t'
                  << ggml_backend_dev_description(dev) << '\n';
    }
}

std::string json_escape(const std::string & input) {
    std::ostringstream out;
    for (unsigned char c : input) {
        switch (c) {
            case '\"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

std::vector<std::string> split_tabs(const std::string & line) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (true) {
        const size_t pos = line.find('\t', start);
        if (pos == std::string::npos) {
            fields.push_back(line.substr(start));
            break;
        }
        fields.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return fields;
}

double sigmoid(double value) {
    if (value >= 0.0) {
        const double z = std::exp(-value);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(value);
    return z / (1.0 + z);
}

double elapsed_ms_since(const std::chrono::steady_clock::time_point & started) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

uint64_t peak_rss_bytes() {
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
    // Linux reports ru_maxrss in KiB.
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
}

enum llama_flash_attn_type flash_attention_type(const std::string & value) {
    if (value == "on") {
        return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    if (value == "auto") {
        return LLAMA_FLASH_ATTN_TYPE_AUTO;
    }
    return LLAMA_FLASH_ATTN_TYPE_DISABLED;
}

std::string make_prompt(
        const std::string & system_prompt,
        const std::string & instruction,
        const std::string & query_text,
        const std::string & document_text) {
    // This is the exact token-bearing structure emitted by the model's official
    // chat template for an image query and a text document. mtmd replaces its
    // generic marker with Qwen's vision_start/image embeddings/vision_end span.
    std::ostringstream prompt;
    prompt << "<|im_start|>system\n" << system_prompt << "<|im_end|>\n"
           << "<|im_start|>user\n"
           << "<Instruct>: " << instruction
           << "<Query>:" << mtmd_default_marker();
    if (!query_text.empty()) {
        prompt << query_text;
    }
    prompt << "\n<Document>:";
    if (document_text.empty()) {
        prompt << "NULL";
    } else {
        prompt << document_text;
    }
    prompt << "<|im_end|>\n<|im_start|>assistant\n";
    return prompt.str();
}

struct batch_guard {
    explicit batch_guard(int32_t n_tokens, int32_t n_seq_max = 1)
        : value(llama_batch_init(n_tokens, 0, n_seq_max)) {}
    ~batch_guard() { llama_batch_free(value); }
    batch_guard(const batch_guard &) = delete;
    batch_guard & operator=(const batch_guard &) = delete;
    llama_batch value;
};

struct component_timings {
    double bitmap_load_ms = 0.0;
    double tokenization_ms = 0.0;
    double system_prefix_prefill_ms = 0.0;
    double visual_encode_projector_ms = 0.0;
    double image_kv_prefill_ms = 0.0;
    double candidate_text_decode_ms = 0.0;
    double rank_pooling_ms = 0.0;
};

struct score_result {
    double score = 0.0;
    float yes_probability = 0.0f;
    float no_probability = 0.0f;
    double margin = 0.0;
    int32_t n_tokens = 0;
    int32_t n_positions = 0;
    int32_t visual_tokens = 0;
    int32_t text_tokens = 0;
    double elapsed_ms = 0.0;
    bool vision_cache_hit = false;
    bool system_prefix_cache_hit = false;
    bool image_kv_cache_hit = false;
    int32_t batch_size = 1;
    uint64_t peak_rss_bytes = 0;
    component_timings timings;
};

struct score_batch_result {
    std::vector<score_result> results;
    double elapsed_ms = 0.0;
    bool vision_cache_hit = false;
    bool system_prefix_cache_hit = false;
    bool image_kv_cache_hit = false;
    int32_t shared_suffix_tokens = 0;
    uint64_t peak_rss_bytes = 0;
    component_timings timings;
};

struct image_prefix_result {
    bool cache_hit = false;
    double visual_encode_projector_ms = 0.0;
    double image_kv_prefill_ms = 0.0;
};

struct prepared_prompt {
    chunks_ptr chunks;
    std::vector<llama_token> prefix_tokens;
    std::vector<llama_token> suffix_tokens;
    const mtmd_input_chunk * image_chunk = nullptr;
    int32_t image_tokens = 0;
    llama_pos image_positions = 0;

    prepared_prompt() : chunks(mtmd_input_chunks_init()) {}
    prepared_prompt(prepared_prompt &&) = default;
    prepared_prompt & operator=(prepared_prompt &&) = default;
    prepared_prompt(const prepared_prompt &) = delete;
    prepared_prompt & operator=(const prepared_prompt &) = delete;
};

class reranker {
public:
    explicit reranker(const options & opts) : opts_(opts) {
        if (!std::filesystem::is_regular_file(opts_.model_path)) {
            throw std::runtime_error("Model file does not exist: " + opts_.model_path);
        }
        if (!std::filesystem::is_regular_file(opts_.mmproj_path)) {
            throw std::runtime_error("mmproj file does not exist: " + opts_.mmproj_path);
        }

        std::vector<ggml_backend_dev_t> npu_devices;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (dev && is_npu_device(dev)) {
                npu_devices.push_back(dev);
            }
        }
        if (opts_.compute != "cpu") {
            if (npu_devices.empty()) {
                throw std::runtime_error("No Hexagon/HTP device is registered; use --list-devices to inspect backends");
            }
            if (opts_.compute == "npu") {
                selected_devices_ = npu_devices;
                selected_devices_.push_back(nullptr);
            }
        }

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = opts_.compute == "cpu" ? 0 : -1;
        if (!selected_devices_.empty()) {
            mparams.devices = selected_devices_.data();
        }
        model_.reset(llama_model_load_from_file(opts_.model_path.c_str(), mparams));
        if (!model_) {
            throw std::runtime_error("Failed to load text model: " + opts_.model_path);
        }
        const uint32_t n_cls_out = llama_model_n_cls_out(model_.get());
        if (n_cls_out != 2) {
            throw std::runtime_error("Expected a two-class reranker GGUF, got " + std::to_string(n_cls_out) + " outputs");
        }
        for (uint32_t i = 0; i < n_cls_out; ++i) {
            const char * label_ptr = llama_model_cls_label(model_.get(), i);
            const std::string label = label_ptr ? lower(label_ptr) : "";
            if (label == "yes") {
                yes_class_index_ = static_cast<int32_t>(i);
            } else if (label == "no") {
                no_class_index_ = static_cast<int32_t>(i);
            }
        }
        if (yes_class_index_ < 0 || no_class_index_ < 0) {
            throw std::runtime_error("Classifier labels must contain both 'yes' and 'no'");
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = static_cast<uint32_t>(opts_.n_ctx);
        cparams.n_batch = static_cast<uint32_t>(opts_.n_batch);
        cparams.n_ubatch = static_cast<uint32_t>(opts_.n_batch);
        cparams.n_seq_max = static_cast<uint32_t>(opts_.max_batch_candidates + 1);
        cparams.n_threads = opts_.n_threads;
        cparams.n_threads_batch = opts_.n_threads;
        cparams.flash_attn_type = flash_attention_type(opts_.flash_attention);
        cparams.embeddings = true;
        cparams.pooling_type = LLAMA_POOLING_TYPE_RANK;
        cparams.kv_unified = true;
        context_.reset(llama_init_from_model(model_.get(), cparams));
        if (!context_) {
            throw std::runtime_error("Failed to create llama context");
        }

        mtmd_context_params vparams = mtmd_context_params_default();
        vparams.use_gpu = opts_.compute != "cpu";
        vparams.print_timings = opts_.print_timings;
        vparams.n_threads = opts_.n_threads;
        vparams.flash_attn_type = flash_attention_type(opts_.flash_attention);
        vparams.warmup = false;
        vparams.image_min_tokens = opts_.image_min_tokens;
        vparams.image_max_tokens = opts_.image_max_tokens;
        vision_.reset(mtmd_init_from_file(opts_.mmproj_path.c_str(), model_.get(), vparams));
        if (!vision_) {
            throw std::runtime_error("Failed to load multimodal projector: " + opts_.mmproj_path);
        }
        if (!mtmd_support_vision(vision_.get())) {
            throw std::runtime_error("The supplied mmproj does not advertise vision support");
        }
    }

    score_result score(
            const std::string & image_path,
            const std::string & query_text,
            const std::string & document_text,
            const std::string & instruction) {
        score_batch_result batch = score_batch(
            image_path, query_text, std::vector<std::string>{document_text}, instruction);
        if (batch.results.size() != 1) {
            throw std::runtime_error("Internal error: a single score did not produce one result");
        }
        score_result result = batch.results.front();
        result.elapsed_ms = batch.elapsed_ms;
        return result;
    }

    score_batch_result score_batch(
            const std::string & image_path,
            const std::string & query_text,
            const std::vector<std::string> & document_texts,
            const std::string & instruction) {
        const auto started = std::chrono::steady_clock::now();
        if (image_path.empty() || !std::filesystem::is_regular_file(image_path)) {
            throw std::runtime_error("Image file does not exist: " + image_path);
        }
        if (document_texts.empty()) {
            throw std::runtime_error("At least one candidate document is required");
        }
        if (document_texts.size() > static_cast<size_t>(opts_.max_batch_candidates)) {
            throw std::runtime_error(
                "Candidate batch size " + std::to_string(document_texts.size()) +
                " exceeds --max-batch-candidates=" + std::to_string(opts_.max_batch_candidates));
        }

        component_timings timings;
        auto component_started = std::chrono::steady_clock::now();
        const bool vision_cache_hit = load_bitmap(image_path);
        timings.bitmap_load_ms = elapsed_ms_since(component_started);

        component_started = std::chrono::steady_clock::now();
        std::vector<prepared_prompt> prompts;
        prompts.reserve(document_texts.size());
        for (const std::string & document : document_texts) {
            prompts.push_back(prepare_prompt(image_path, query_text, document, instruction));
        }
        validate_prompt_batch(prompts);
        timings.tokenization_ms = elapsed_ms_since(component_started);

        component_started = std::chrono::steady_clock::now();
        const bool system_prefix_cache_hit = ensure_system_prefix(prompts.front().prefix_tokens);
        timings.system_prefix_prefill_ms = elapsed_ms_since(component_started);
        const image_prefix_result image_prefix = ensure_image_prefix(
            image_path, prompts.front().image_chunk);
        timings.visual_encode_projector_ms = image_prefix.visual_encode_projector_ms;
        timings.image_kv_prefill_ms = image_prefix.image_kv_prefill_ms;
        const size_t shared_suffix_tokens = common_suffix_prefix_length(prompts);
        const llama_pos image_prefix_n_past = image_n_past_;

        clear_candidate_sequences();
        try {
            if (shared_suffix_tokens > 0) {
                component_started = std::chrono::steady_clock::now();
                decode_text_span(
                    prompts.front().suffix_tokens, 0, shared_suffix_tokens, cache_sequence_id(),
                    image_n_past_, text_batch_limit(), false);
                timings.candidate_text_decode_ms += elapsed_ms_since(component_started);
            }

            llama_memory_t memory = llama_get_memory(context_.get());
            for (size_t i = 0; i < prompts.size(); ++i) {
                llama_memory_seq_cp(memory, cache_sequence_id(), candidate_sequence_id(i), -1, -1);
            }

            std::vector<score_result> results = decode_candidate_suffixes(
                prompts, shared_suffix_tokens,
                image_prefix_n_past + static_cast<llama_pos>(shared_suffix_tokens),
                timings);
            clear_candidate_sequences();
            trim_shared_suffix();

            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            const double amortized_ms = elapsed_ms / static_cast<double>(results.size());
            for (size_t i = 0; i < results.size(); ++i) {
                score_result & result = results[i];
                result.n_tokens = static_cast<int32_t>(
                    prompts[i].prefix_tokens.size() + prompts[i].image_tokens + prompts[i].suffix_tokens.size());
                result.visual_tokens = prompts[i].image_tokens;
                result.text_tokens = static_cast<int32_t>(
                    prompts[i].prefix_tokens.size() + prompts[i].suffix_tokens.size());
                result.n_positions = image_prefix_n_past +
                                     static_cast<int32_t>(prompts[i].suffix_tokens.size());
                result.elapsed_ms = amortized_ms;
                result.vision_cache_hit = vision_cache_hit || i > 0;
                result.system_prefix_cache_hit = system_prefix_cache_hit;
                result.image_kv_cache_hit = image_prefix.cache_hit || i > 0;
                result.batch_size = static_cast<int32_t>(results.size());
                result.peak_rss_bytes = peak_rss_bytes();
                result.timings = timings;
            }

            score_batch_result batch;
            batch.results = std::move(results);
            batch.elapsed_ms = elapsed_ms;
            batch.vision_cache_hit = vision_cache_hit;
            batch.system_prefix_cache_hit = system_prefix_cache_hit;
            batch.image_kv_cache_hit = image_prefix.cache_hit;
            batch.shared_suffix_tokens = static_cast<int32_t>(shared_suffix_tokens);
            batch.peak_rss_bytes = peak_rss_bytes();
            batch.timings = timings;
            return batch;
        } catch (...) {
            clear_candidate_sequences();
            trim_shared_suffix();
            throw;
        }
    }

    score_result score_uncached_reference(
            const std::string & image_path,
            const std::string & query_text,
            const std::string & document_text,
            const std::string & instruction) {
        const auto started = std::chrono::steady_clock::now();
        if (image_path.empty() || !std::filesystem::is_regular_file(image_path)) {
            throw std::runtime_error("Image file does not exist: " + image_path);
        }

        bool cache_hit = false;
        if (cached_image_path_ != image_path || !cached_bitmap_) {
            auto loaded = mtmd_helper_bitmap_init_from_file(vision_.get(), image_path.c_str(), false);
            if (!loaded.bitmap) {
                throw std::runtime_error("Failed to decode image: " + image_path);
            }
            // A video context is not expected for an image reranker and would
            // need a different lifetime API, so reject it explicitly.
            if (loaded.video_ctx) {
                mtmd_helper_video_free(loaded.video_ctx);
                mtmd_bitmap_free(loaded.bitmap);
                throw std::runtime_error("Video input is not supported by this image reranker executable");
            }
            cached_bitmap_.reset(loaded.bitmap);
            cached_image_path_ = image_path;
            cached_vision_embd_.clear();
            cached_vision_tokens_ = 0;
        } else {
            cache_hit = !cached_vision_embd_.empty();
        }

        llama_memory_clear(llama_get_memory(context_.get()), true);

        const std::string prompt = make_prompt(opts_.system_prompt, instruction, query_text, document_text);
        mtmd_input_text input_text {
            /*.text          =*/ prompt.c_str(),
            /*.add_special   =*/ false,
            /*.parse_special =*/ true,
        };
        chunks_ptr chunks(mtmd_input_chunks_init());
        const mtmd_bitmap * bitmaps[] = {cached_bitmap_.get()};
        const int32_t tokenize_rc = mtmd_tokenize(vision_.get(), chunks.get(), &input_text, bitmaps, 1);
        if (tokenize_rc != 0) {
            throw std::runtime_error("mtmd_tokenize failed with code " + std::to_string(tokenize_rc));
        }

        const size_t n_chunks = mtmd_input_chunks_size(chunks.get());
        if (n_chunks == 0) {
            throw std::runtime_error("mtmd_tokenize returned no chunks");
        }
        llama_pos n_past = 0;
        int32_t total_tokens = 0;
        for (size_t i = 0; i < n_chunks; ++i) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.get(), i);
            const auto chunk_type = mtmd_input_chunk_get_type(chunk);
            const int32_t chunk_tokens = static_cast<int32_t>(mtmd_input_chunk_get_n_tokens(chunk));
            total_tokens += chunk_tokens;
            llama_pos new_n_past = n_past;
            int32_t rc = 0;

            if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                // GenieX v0.3.13's Hexagon backend can wait forever when the
                // final rank-pooling text graph is evaluated as one larger
                // micro-batch (reproduced at 134 positions). Split only the
                // final text chunk on HTP; keep the image chunk at n_batch so
                // vision latency and its cached embedding path are unchanged.
                const int32_t text_batch =
                    opts_.compute == "cpu" || i != n_chunks - 1
                        ? opts_.n_batch
                        : std::min<int32_t>(opts_.n_batch, 32);
                rc = mtmd_helper_eval_chunk_single(
                    vision_.get(), context_.get(), chunk, n_past, 0, text_batch,
                    i == n_chunks - 1, &new_n_past);
            } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                const size_t embd_count = static_cast<size_t>(llama_model_n_embd_inp(model_.get())) *
                                          static_cast<size_t>(chunk_tokens);
                if (cached_vision_embd_.empty()) {
                    rc = mtmd_encode_chunk(vision_.get(), chunk);
                    if (rc == 0) {
                        const float * output = mtmd_get_output_embd(vision_.get());
                        if (!output) {
                            throw std::runtime_error("mtmd returned a null vision embedding");
                        }
                        cached_vision_embd_.assign(output, output + embd_count);
                        cached_vision_tokens_ = chunk_tokens;
                    }
                } else if (cached_vision_tokens_ != chunk_tokens || cached_vision_embd_.size() != embd_count) {
                    throw std::runtime_error("Cached vision embedding shape changed for the same image");
                }
                if (rc == 0) {
                    rc = mtmd_helper_decode_image_chunk(
                        vision_.get(), context_.get(), chunk, cached_vision_embd_.data(),
                        n_past, 0, opts_.n_batch, &new_n_past, nullptr, nullptr);
                }
            } else {
                throw std::runtime_error("Audio chunks are not supported by this reranker");
            }
            if (rc != 0) {
                throw std::runtime_error("Failed to evaluate mtmd chunk " + std::to_string(i) +
                                         " with code " + std::to_string(rc));
            }
            n_past = new_n_past;
        }

        float * probabilities = llama_get_embeddings_seq(context_.get(), 0);
        if (!probabilities) {
            throw std::runtime_error("llama_get_embeddings_seq returned null for the rank-pooled sequence");
        }
        score_result result;
        result.yes_probability = probabilities[yes_class_index_];
        result.no_probability = probabilities[no_class_index_];
        result.score = result.yes_probability;
        result.margin = std::log(std::max<double>(result.yes_probability, 1e-30)) -
                        std::log(std::max<double>(result.no_probability, 1e-30));
        result.n_tokens = total_tokens;
        result.n_positions = n_past;
        result.vision_cache_hit = cache_hit;
        result.elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return result;
    }

private:
    llama_seq_id cache_sequence_id() const {
        return static_cast<llama_seq_id>(opts_.max_batch_candidates);
    }

    static llama_seq_id candidate_sequence_id(size_t candidate_index) {
        return static_cast<llama_seq_id>(candidate_index);
    }

    int32_t text_batch_limit() const {
        return opts_.compute == "cpu" ? opts_.n_batch : std::min<int32_t>(opts_.n_batch, 32);
    }

    static void batch_add_token(
            llama_batch & batch,
            llama_token token,
            llama_pos position,
            llama_seq_id seq_id,
            bool output) {
        const int32_t index = batch.n_tokens++;
        batch.token[index] = token;
        batch.pos[index] = position;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = seq_id;
        batch.logits[index] = output;
    }

    void decode_text_span(
            const std::vector<llama_token> & tokens,
            size_t begin,
            size_t end,
            llama_seq_id seq_id,
            llama_pos start_position,
            int32_t batch_limit,
            bool output_last) {
        if (begin > end || end > tokens.size()) {
            throw std::runtime_error("Invalid text token span");
        }
        if (begin == end) {
            return;
        }
        batch_guard holder(batch_limit);
        size_t cursor = begin;
        while (cursor < end) {
            llama_batch & batch = holder.value;
            batch.n_tokens = 0;
            while (cursor < end && batch.n_tokens < batch_limit) {
                const bool is_last = output_last && cursor + 1 == end;
                batch_add_token(
                    batch, tokens[cursor],
                    start_position + static_cast<llama_pos>(cursor - begin),
                    seq_id, is_last);
                ++cursor;
            }
            const int32_t rc = llama_decode(context_.get(), batch);
            if (rc != 0) {
                throw std::runtime_error("Failed to decode text batch with code " + std::to_string(rc));
            }
        }
    }

    void invalidate_all_prefixes() {
        llama_memory_clear(llama_get_memory(context_.get()), true);
        base_prefix_valid_ = false;
        image_prefix_valid_ = false;
        base_prefix_tokens_.clear();
        base_n_past_ = 0;
        image_n_past_ = 0;
        image_prefix_path_.clear();
    }

    void clear_candidate_sequences() {
        llama_memory_t memory = llama_get_memory(context_.get());
        for (size_t i = 0; i < static_cast<size_t>(opts_.max_batch_candidates); ++i) {
            llama_memory_seq_rm(memory, candidate_sequence_id(i), -1, -1);
        }
    }

    void trim_shared_suffix() {
        if (!image_prefix_valid_) {
            return;
        }
        if (!llama_memory_seq_rm(
                llama_get_memory(context_.get()), cache_sequence_id(), image_n_past_, -1)) {
            invalidate_all_prefixes();
        }
    }

    void drop_image_prefix() {
        clear_candidate_sequences();
        if (image_prefix_valid_ &&
            !llama_memory_seq_rm(
                llama_get_memory(context_.get()), cache_sequence_id(), base_n_past_, -1)) {
            invalidate_all_prefixes();
            return;
        }
        image_prefix_valid_ = false;
        image_n_past_ = base_n_past_;
        image_prefix_path_.clear();
    }

    bool load_bitmap(const std::string & image_path) {
        const uintmax_t image_size = std::filesystem::file_size(image_path);
        const auto image_mtime = std::filesystem::last_write_time(image_path);
        if (cached_image_identity_valid_ && cached_image_path_ == image_path && cached_bitmap_ &&
            cached_image_size_ == image_size && cached_image_mtime_ == image_mtime) {
            return !cached_vision_embd_.empty();
        }

        drop_image_prefix();
        auto loaded = mtmd_helper_bitmap_init_from_file(vision_.get(), image_path.c_str(), false);
        if (!loaded.bitmap) {
            throw std::runtime_error("Failed to decode image: " + image_path);
        }
        if (loaded.video_ctx) {
            mtmd_helper_video_free(loaded.video_ctx);
            mtmd_bitmap_free(loaded.bitmap);
            throw std::runtime_error("Video input is not supported by this image reranker executable");
        }
        cached_bitmap_.reset(loaded.bitmap);
        cached_image_path_ = image_path;
        cached_image_size_ = image_size;
        cached_image_mtime_ = image_mtime;
        cached_image_identity_valid_ = true;
        cached_vision_embd_.clear();
        cached_vision_tokens_ = 0;
        return false;
    }

    prepared_prompt prepare_prompt(
            const std::string & image_path,
            const std::string & query_text,
            const std::string & document_text,
            const std::string & instruction) {
        prepared_prompt prepared;
        if (!prepared.chunks) {
            throw std::runtime_error("Failed to allocate mtmd input chunks");
        }
        const std::string prompt = make_prompt(opts_.system_prompt, instruction, query_text, document_text);
        mtmd_input_text input_text {
            /*.text          =*/ prompt.c_str(),
            /*.add_special   =*/ false,
            /*.parse_special =*/ true,
        };
        const mtmd_bitmap * bitmaps[] = {cached_bitmap_.get()};
        const int32_t rc = mtmd_tokenize(vision_.get(), prepared.chunks.get(), &input_text, bitmaps, 1);
        if (rc != 0) {
            throw std::runtime_error(
                "mtmd_tokenize failed for " + image_path + " with code " + std::to_string(rc));
        }

        const size_t n_chunks = mtmd_input_chunks_size(prepared.chunks.get());
        bool seen_image = false;
        for (size_t i = 0; i < n_chunks; ++i) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(prepared.chunks.get(), i);
            const auto type = mtmd_input_chunk_get_type(chunk);
            if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n_tokens = 0;
                const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
                std::vector<llama_token> & destination =
                    seen_image ? prepared.suffix_tokens : prepared.prefix_tokens;
                destination.insert(destination.end(), tokens, tokens + n_tokens);
            } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                if (seen_image) {
                    throw std::runtime_error("Only one image chunk is supported per reranker request");
                }
                seen_image = true;
                prepared.image_chunk = chunk;
                prepared.image_tokens = static_cast<int32_t>(mtmd_input_chunk_get_n_tokens(chunk));
                prepared.image_positions = mtmd_input_chunk_get_n_pos(chunk);
            } else {
                throw std::runtime_error("Audio chunks are not supported by this reranker");
            }
        }
        if (!prepared.image_chunk || prepared.prefix_tokens.empty() || prepared.suffix_tokens.empty()) {
            throw std::runtime_error("Unexpected mtmd prompt layout; expected text, image, then text");
        }
        return prepared;
    }

    void validate_prompt_batch(const std::vector<prepared_prompt> & prompts) const {
        const prepared_prompt & first = prompts.front();
        for (size_t i = 1; i < prompts.size(); ++i) {
            if (prompts[i].prefix_tokens != first.prefix_tokens) {
                throw std::runtime_error("Candidate prompts do not share an identical system/image prefix");
            }
            if (prompts[i].image_tokens != first.image_tokens ||
                prompts[i].image_positions != first.image_positions) {
                throw std::runtime_error("Candidate image token shapes do not match");
            }
        }
    }

    bool ensure_system_prefix(const std::vector<llama_token> & prefix_tokens) {
        if (base_prefix_valid_ && prefix_tokens == base_prefix_tokens_) {
            return true;
        }
        invalidate_all_prefixes();
        decode_text_span(prefix_tokens, 0, prefix_tokens.size(), cache_sequence_id(), 0, opts_.n_batch, false);
        base_prefix_tokens_ = prefix_tokens;
        base_n_past_ = static_cast<llama_pos>(prefix_tokens.size());
        image_n_past_ = base_n_past_;
        base_prefix_valid_ = true;
        return false;
    }

    image_prefix_result ensure_image_prefix(
            const std::string & image_path,
            const mtmd_input_chunk * image_chunk) {
        image_prefix_result result;
        if (image_prefix_valid_ && image_prefix_path_ == image_path) {
            result.cache_hit = true;
            return result;
        }
        drop_image_prefix();
        if (!base_prefix_valid_) {
            throw std::runtime_error("System prefix is not initialized");
        }

        const int32_t image_tokens = static_cast<int32_t>(mtmd_input_chunk_get_n_tokens(image_chunk));
        const size_t embedding_count = static_cast<size_t>(llama_model_n_embd_inp(model_.get())) *
                                       static_cast<size_t>(image_tokens);
        if (cached_vision_embd_.empty()) {
            const auto encode_started = std::chrono::steady_clock::now();
            const int32_t encode_rc = mtmd_encode_chunk(vision_.get(), image_chunk);
            if (encode_rc != 0) {
                throw std::runtime_error("Failed to encode image with code " + std::to_string(encode_rc));
            }
            const float * output = mtmd_get_output_embd(vision_.get());
            if (!output) {
                throw std::runtime_error("mtmd returned a null vision embedding");
            }
            cached_vision_embd_.assign(output, output + embedding_count);
            cached_vision_tokens_ = image_tokens;
            result.visual_encode_projector_ms = elapsed_ms_since(encode_started);
        } else if (cached_vision_tokens_ != image_tokens || cached_vision_embd_.size() != embedding_count) {
            throw std::runtime_error("Cached vision embedding shape changed for the same image");
        }

        llama_pos new_n_past = base_n_past_;
        const auto prefill_started = std::chrono::steady_clock::now();
        const int32_t decode_rc = mtmd_helper_decode_image_chunk(
            vision_.get(), context_.get(), image_chunk, cached_vision_embd_.data(),
            base_n_past_, cache_sequence_id(), opts_.n_batch, &new_n_past, nullptr, nullptr);
        if (decode_rc != 0) {
            throw std::runtime_error("Failed to decode image prefix with code " + std::to_string(decode_rc));
        }
        result.image_kv_prefill_ms = elapsed_ms_since(prefill_started);
        image_n_past_ = new_n_past;
        image_prefix_path_ = image_path;
        image_prefix_valid_ = true;
        return result;
    }

    static size_t common_suffix_prefix_length(const std::vector<prepared_prompt> & prompts) {
        size_t common = prompts.front().suffix_tokens.size() - 1;
        for (size_t i = 1; i < prompts.size(); ++i) {
            common = std::min(common, prompts[i].suffix_tokens.size() - 1);
        }
        for (size_t token_index = 0; token_index < common; ++token_index) {
            const llama_token token = prompts.front().suffix_tokens[token_index];
            for (size_t i = 1; i < prompts.size(); ++i) {
                if (prompts[i].suffix_tokens[token_index] != token) {
                    return token_index;
                }
            }
        }
        return prompts.size() > 1 ? common : 0;
    }

    std::vector<score_result> decode_candidate_suffixes(
            const std::vector<prepared_prompt> & prompts,
            size_t shared_prefix_tokens,
            llama_pos suffix_start_position,
            component_timings & timings) {
        const size_t candidate_count = prompts.size();
        std::vector<size_t> cursors(candidate_count, shared_prefix_tokens);
        const int32_t batch_limit = text_batch_limit();
        std::vector<score_result> results(candidate_count);
        const auto capture_score = [&](size_t candidate_index) {
            float * probabilities = llama_get_embeddings_seq(
                context_.get(), candidate_sequence_id(candidate_index));
            if (!probabilities) {
                throw std::runtime_error(
                    "llama_get_embeddings_seq returned null for candidate " +
                    std::to_string(candidate_index));
            }
            score_result & result = results[candidate_index];
            result.yes_probability = probabilities[yes_class_index_];
            result.no_probability = probabilities[no_class_index_];
            result.score = result.yes_probability;
            result.margin = std::log(std::max<double>(result.yes_probability, 1e-30)) -
                            std::log(std::max<double>(result.no_probability, 1e-30));
        };

        // Preserve the exact historical single-candidate graph shape. This is
        // also useful for callers that deliberately request only one score.
        if (candidate_count == 1) {
            const auto & tokens = prompts.front().suffix_tokens;
            if (shared_prefix_tokens + 1 < tokens.size()) {
                const auto text_started = std::chrono::steady_clock::now();
                decode_text_span(
                    tokens, shared_prefix_tokens, tokens.size() - 1, candidate_sequence_id(0),
                    suffix_start_position, batch_limit, false);
                timings.candidate_text_decode_ms += elapsed_ms_since(text_started);
            }
            const auto rank_started = std::chrono::steady_clock::now();
            batch_guard final_holder(1);
            llama_batch & final_batch = final_holder.value;
            final_batch.n_tokens = 0;
            batch_add_token(
                final_batch, tokens.back(),
                suffix_start_position +
                    static_cast<llama_pos>(tokens.size() - 1 - shared_prefix_tokens),
                candidate_sequence_id(0), true);
            const int32_t rc = llama_decode(context_.get(), final_batch);
            if (rc != 0) {
                throw std::runtime_error(
                    "Failed to decode final candidate token with code " + std::to_string(rc));
            }
            capture_score(0);
            timings.rank_pooling_ms += elapsed_ms_since(rank_started);
            return results;
        }

        {
            const auto text_started = std::chrono::steady_clock::now();
            batch_guard body_holder(batch_limit);
            while (true) {
                llama_batch & batch = body_holder.value;
                batch.n_tokens = 0;
                bool added = false;
                const int32_t per_candidate = std::max<int32_t>(
                    1, batch_limit / static_cast<int32_t>(candidate_count));
                for (size_t i = 0; i < candidate_count && batch.n_tokens < batch_limit; ++i) {
                    int32_t added_for_candidate = 0;
                    const auto & tokens = prompts[i].suffix_tokens;
                    while (cursors[i] + 1 < tokens.size() &&
                           batch.n_tokens < batch_limit &&
                           added_for_candidate < per_candidate) {
                        batch_add_token(
                            batch, tokens[cursors[i]],
                            suffix_start_position +
                                static_cast<llama_pos>(cursors[i] - shared_prefix_tokens),
                            candidate_sequence_id(i), false);
                        ++cursors[i];
                        ++added_for_candidate;
                        added = true;
                    }
                }
                if (!added) {
                    break;
                }
                const int32_t rc = llama_decode(context_.get(), batch);
                if (rc != 0) {
                    throw std::runtime_error(
                        "Failed to decode candidate batch with code " + std::to_string(rc));
                }
            }
            timings.candidate_text_decode_ms += elapsed_ms_since(text_started);
        }

        for (size_t i = 0; i < candidate_count; ++i) {
            if (cursors[i] + 1 != prompts[i].suffix_tokens.size()) {
                throw std::runtime_error("Candidate suffix cursor did not reach its final token");
            }
        }

        // Capture each output group before the next decode because llama.cpp
        // clears rank-pooled sequence outputs per decode. GenieX/HTP is capped
        // at four shared-prefix candidates by argument validation; CPU callers
        // may explicitly request a larger batch.
        const size_t final_group_limit = opts_.compute == "cpu" ? candidate_count : 4;
        const auto rank_started = std::chrono::steady_clock::now();
        for (size_t begin = 0; begin < candidate_count; begin += final_group_limit) {
            const size_t end = std::min(candidate_count, begin + final_group_limit);
            batch_guard final_holder(static_cast<int32_t>(end - begin));
            llama_batch & final_batch = final_holder.value;
            final_batch.n_tokens = 0;
            for (size_t i = begin; i < end; ++i) {
                const auto & tokens = prompts[i].suffix_tokens;
                batch_add_token(
                    final_batch, tokens.back(),
                    suffix_start_position +
                        static_cast<llama_pos>(tokens.size() - 1 - shared_prefix_tokens),
                    candidate_sequence_id(i), true);
            }
            const int32_t rc = llama_decode(context_.get(), final_batch);
            if (rc != 0) {
                throw std::runtime_error(
                    "Failed to decode final candidate batch with code " + std::to_string(rc));
            }
            for (size_t i = begin; i < end; ++i) {
                capture_score(i);
            }
        }
        timings.rank_pooling_ms += elapsed_ms_since(rank_started);
        return results;
    }

    options opts_;
    std::vector<ggml_backend_dev_t> selected_devices_;
    model_ptr model_;
    context_ptr context_;
    mtmd_ptr vision_;
    std::string cached_image_path_;
    bitmap_ptr cached_bitmap_;
    uintmax_t cached_image_size_ = 0;
    std::filesystem::file_time_type cached_image_mtime_ {};
    bool cached_image_identity_valid_ = false;
    std::vector<float> cached_vision_embd_;
    int32_t cached_vision_tokens_ = 0;
    std::vector<llama_token> base_prefix_tokens_;
    std::string image_prefix_path_;
    llama_pos base_n_past_ = 0;
    llama_pos image_n_past_ = 0;
    bool base_prefix_valid_ = false;
    bool image_prefix_valid_ = false;
    int32_t yes_class_index_ = -1;
    int32_t no_class_index_ = -1;
};

void print_component_timings(std::ostream & out, const component_timings & timings) {
    out << "{\"bitmap_load_ms\":" << timings.bitmap_load_ms
        << ",\"tokenization_ms\":" << timings.tokenization_ms
        << ",\"system_prefix_prefill_ms\":" << timings.system_prefix_prefill_ms
        << ",\"visual_encode_projector_ms\":" << timings.visual_encode_projector_ms
        << ",\"image_kv_prefill_ms\":" << timings.image_kv_prefill_ms
        << ",\"candidate_text_decode_ms\":" << timings.candidate_text_decode_ms
        << ",\"rank_pooling_ms\":" << timings.rank_pooling_ms << "}";
}

void print_backend_metadata(std::ostream & out, const std::string & compute) {
    const std::string text_backend = compute == "npu"
        ? "htp_pinned"
        : (compute == "hybrid" ? "hybrid" : "cpu");
    const std::string visual_backend = compute == "cpu" ? "cpu" : "opencl_gpu";
    out << ",\"backend_placement\":{\"text\":\"" << text_backend
        << "\",\"visual_encoder_projector\":\"" << visual_backend << "\"}"
        << ",\"fallback_operations\":[";
    if (compute != "cpu") {
        // GenieX 0.3.13 reports this mtmd operation on CPU. Keeping it in the
        // structured trace makes removal by deterministic preprocessing visible.
        out << "\"vision.UPSCALE:cpu\"";
    }
    out << "]";
}

void print_result_object(
        std::ostream & out,
        const score_result & result,
        const std::string & image,
        const std::string & document,
        const std::string & compute) {
    out << std::setprecision(10)
        << "{\"score\":" << result.score
        << ",\"margin_log_odds\":" << result.margin
        << ",\"yes_probability\":" << result.yes_probability
        << ",\"no_probability\":" << result.no_probability
        << ",\"tokens\":" << result.n_tokens
        << ",\"positions\":" << result.n_positions
        << ",\"visual_tokens\":" << result.visual_tokens
        << ",\"text_tokens\":" << result.text_tokens
        << ",\"elapsed_ms\":" << result.elapsed_ms
        << ",\"vision_cache_hit\":" << (result.vision_cache_hit ? "true" : "false")
        << ",\"system_prefix_cache_hit\":" << (result.system_prefix_cache_hit ? "true" : "false")
        << ",\"image_kv_cache_hit\":" << (result.image_kv_cache_hit ? "true" : "false")
        << ",\"batch_size\":" << result.batch_size
        << ",\"peak_rss_bytes\":" << result.peak_rss_bytes
        << ",\"timings_ms\":";
    print_component_timings(out, result.timings);
    print_backend_metadata(out, compute);
    out
        << ",\"compute\":\"" << json_escape(compute) << "\""
        << ",\"image\":\"" << json_escape(image) << "\""
        << ",\"document\":\"" << json_escape(document) << "\"}";
}

void print_result(
        const score_result & result,
        const std::string & image,
        const std::string & document,
        const std::string & compute) {
    print_result_object(std::cout, result, image, document, compute);
    std::cout << std::endl;
}

void print_batch_result(
        const score_batch_result & batch,
        const std::string & image,
        const std::vector<std::string> & documents,
        const std::string & compute) {
    std::cout << std::setprecision(10) << "{\"results\":[";
    for (size_t i = 0; i < batch.results.size(); ++i) {
        if (i > 0) {
            std::cout << ',';
        }
        print_result_object(std::cout, batch.results[i], image, documents[i], compute);
    }
    std::cout << "],\"elapsed_ms\":" << batch.elapsed_ms
              << ",\"batch_size\":" << batch.results.size()
              << ",\"shared_suffix_tokens\":" << batch.shared_suffix_tokens
              << ",\"vision_cache_hit\":" << (batch.vision_cache_hit ? "true" : "false")
              << ",\"system_prefix_cache_hit\":" << (batch.system_prefix_cache_hit ? "true" : "false")
              << ",\"image_kv_cache_hit\":" << (batch.image_kv_cache_hit ? "true" : "false")
              << ",\"peak_rss_bytes\":" << batch.peak_rss_bytes
              << ",\"timings_ms\":";
    print_component_timings(std::cout, batch.timings);
    print_backend_metadata(std::cout, compute);
    std::cout
              << "}" << std::endl;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        options opts = parse_args(argc, argv);
        if (!opts.verbose) {
            // Model and projector loading emits thousands of informational
            // lines. Keep the JSON protocol clean by default; --verbose is
            // available for placement and backend diagnostics.
            llama_log_set(warning_log_callback, nullptr);
        }
        // FastRPC must be able to find GenieX's libggml-htp-v*.so skeleton.
        // QNN/ONNX Runtime may already have set ADSP_LIBRARY_PATH to only its
        // Python package directory. Always put the GenieX paths first while
        // preserving the inherited entries needed by other Qualcomm stacks.
        const char * inherited_adsp = std::getenv("ADSP_LIBRARY_PATH");
        std::string adsp_path = opts.backend_dir + ";/usr/lib/dsp/cdsp";
        if (inherited_adsp && inherited_adsp[0] != '\0') {
            adsp_path += ";";
            adsp_path += inherited_adsp;
        }
        setenv("ADSP_LIBRARY_PATH", adsp_path.c_str(), 1);
        backend_guard backend;
        ggml_backend_load_all_from_path(opts.backend_dir.c_str());

        if (opts.list_devices) {
            print_devices();
            return 0;
        }
        if (opts.model_path.empty() || opts.mmproj_path.empty()) {
            throw std::runtime_error("--model and --mmproj are required");
        }
        if (!opts.interactive && (opts.image_path.empty() || opts.document_text.empty())) {
            throw std::runtime_error("--image and --document are required unless --interactive is used");
        }

        reranker engine(opts);
        if (!opts.interactive) {
            const score_result result = engine.score(
                opts.image_path, opts.query_text, opts.document_text, opts.instruction);
            print_result(result, opts.image_path, opts.document_text, opts.compute);
        } else {
            // Let persistent clients distinguish model/projector load time
            // from the first scored pair and fail early if initialization did
            // not complete.
            std::cout << "{\"ready\":true,\"compute\":\"" << json_escape(opts.compute)
                      << "\",\"batch_protocol\":true,\"max_batch_candidates\":"
                      << opts.max_batch_candidates
                      << ",\"n_ctx\":" << opts.n_ctx
                      << ",\"n_batch\":" << opts.n_batch
                      << ",\"threads\":" << opts.n_threads
                      << ",\"image_min_tokens\":" << opts.image_min_tokens
                      << ",\"image_max_tokens\":" << opts.image_max_tokens
                      << ",\"flash_attention\":\"" << json_escape(opts.flash_attention) << "\""
                      << ",\"individual_call_protocol\":"
                      << (opts.max_batch_candidates == 1 ? "true" : "false");
            print_backend_metadata(std::cout, opts.compute);
            std::cout << "}" << std::endl;
            // Legacy protocol: image<TAB>query<TAB>document[<TAB>instruction].
            // Batch protocol: BATCH<TAB>image<TAB>query<TAB>instruction<TAB>document...
            // Fields must not contain literal tabs/newlines. One JSON object is
            // emitted per request.
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line.empty()) {
                    continue;
                }
                try {
                    const auto fields = split_tabs(line);
                    if (!fields.empty() && fields[0] == "BATCH") {
                        if (fields.size() < 5) {
                            throw std::runtime_error("Expected BATCH, image, query, instruction, and document fields");
                        }
                        const std::string & instruction = fields[3].empty() ? opts.instruction : fields[3];
                        const std::vector<std::string> documents(fields.begin() + 4, fields.end());
                        const score_batch_result batch = engine.score_batch(
                            fields[1], fields[2], documents, instruction);
                        print_batch_result(batch, fields[1], documents, opts.compute);
                    } else {
                        if (fields.size() < 3 || fields.size() > 4) {
                            throw std::runtime_error("Expected 3 or 4 tab-separated fields");
                        }
                        const std::string & instruction = fields.size() == 4 && !fields[3].empty()
                            ? fields[3] : opts.instruction;
                        const score_result result = engine.score(fields[0], fields[1], fields[2], instruction);
                        print_result(result, fields[0], fields[2], opts.compute);
                    }
                } catch (const std::exception & exc) {
                    std::cout << "{\"error\":\"" << json_escape(exc.what()) << "\"}" << std::endl;
                }
            }
        }
        return 0;
    } catch (const std::exception & exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
