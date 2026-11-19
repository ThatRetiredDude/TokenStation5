# Models tested on PS5

Every model here ran on a PS5 with Linux (40 GPU compute units, GPU memory set to max) on the Linux edition, and was checked against ts5's CPU reference (all pass except two Gemma cases noted below). Speeds are tokens per second, one request at a time, with no helper: **prompt** is reading a 512-token prompt, **writing** is generating the answer. **Context** is what `-c auto` gives with the current defaults. All files are Q4_K_M unless named.

The [terminal app](Using-the-terminal-app) downloads every model on this page for you: **F2**, pick it under **⇣ Download**, **Enter**, or type `/download qwen3-14b` (the keys are in [its command list](Using-the-terminal-app#commands)). The desktop app's **Models** page downloads the checked models, and its **Search Hugging Face** finds others. The commands below install these exact files from a shell instead.

## The models

| Model | Good for | File size | Prompt | Writing | Context | Helper to pair with it |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Qwen3 14B | Best all-rounder: writing, explaining, reasoning; can think first | 9.0 GB | ~490 | 41 | 32k | Qwen3 0.6B |
| Qwen3.5 9B (DeepSeek-V4-Pro heretic) | Very long documents (found 10/10 facts at 225k tokens); fast follow-ups on the same document. Community fine-tune, uncensored | 5.6 GB | ~826 | 67 | 256k | none |
| Qwen3.8 9B (distill) | Long documents on newer training; thinks, calls tools. A community distill of Qwen3.8 into a 9B | 5.8 GB | ~651 | 67 | 256k | none |
| DeepSeek R1 Distill Qwen 14B (abliterated v2) | Math, logic, puzzles; thinks out loud before answering. Refusals removed | 9.0 GB | ~490 | 38-41 | 32k | Qwen3 0.6B |
| Qwen3 8B | Fast general chat, thinking, tools | 5.0 GB | ~850 | 72 | 32k | Qwen3 0.6B |
| Llama 3.1 8B Instruct | Fast general chat with a long memory (10/10 facts at 111k) | 4.9 GB | ~850 | 74 | 128k | Llama 3.2 1B |
| Qwen 2.5 Coder 7B Instruct | Coding, fastest | 4.7 GB | ~960 | 77 | 32k | Qwen3 0.6B |
| Qwen 2.5 Coder 14B Instruct | Coding, stronger | 9.0 GB | not measured | ~28-30 (long prompts) | 32k | Qwen3 0.6B |
| Llama 3.2 3B Instruct | Quick answers; the fastest Q4_K_M model here | 2.0 GB | ~1,290 | 145 | 128k | Llama 3.2 1B |
| Gemma 3 12B | General chat; weak on long documents past ~8k (4/10 at 53k). One known near-tie in its check | 7.3 GB | ~406 | 45 | 64k | Gemma 3 1B, by tuning only |
| Gemma 4 12B | General chat; 6/10 facts at 112k. Its English-text check is over the bar (being looked into) | 7.7 GB | ~394 | 43 | 128k | Gemma 3 1B, by tuning only |
| Ternary Bonsai 2 27B (Q2_0) | The biggest model that fits: a 27B at two bits a weight. Slower; thinks, calls tools | 7.6 GB | ~229 | 23 | 150k | none |
| Ternary Bonsai 2 9B (Q2_0) | Small file, long memory; thinks, calls tools | 2.6 GB | ~740 | 59 | 256k | none |
| Ternary Bonsai 2 4B (Q2_0) | The fastest here and the smallest download; thinks, calls tools | 1.2 GB | ~1,380 | 119 | 256k | none |

Which one: **Qwen3 14B** for the best answers, **Qwen3.5 9B** for very long documents, **Llama 3.1 8B** or **Qwen3 8B** when speed matters most, **Llama 3.2 3B** for quick answers, **Qwen 2.5 Coder** for code, a **Bonsai** when the download or the memory has to stay small. Long-document scores count 10 facts hidden in a document. More numbers are in [Benchmarks](Benchmarks) and [Test results on PS5 Linux](Test-results-on-PS5-Linux).

**The Bonsai models** ([prism-ml](https://huggingface.co/prism-ml)'s Ternary-Bonsai-2: the 27B built on Qwen3.8 27B, the 9B and 4B on Qwen3.5) keep every weight as −1, 0 or +1, two bits each, so a 27B is a 7.6 GB file and a 4B is 1.2 GB. ts5 reads their files as they are, and all three pass the check against the CPU reference. They think before they answer and call tools like the Qwen models. Use the **Q2_0** file of a repo; the other packings there (PQ2_0, PTQ1_0) need the makers' own program.

**Helpers** are small models of the same family that draft a few words ahead for the big one, which then checks them. That makes writing faster with exactly the same text. The helper in the column is the one tuning picked on the test PS5. It's built into ts5 for these models: each uses it without any tuning once the helper's file is in the same folder (the terminal app downloads it right after the model). The Gemma 12Bs are the exception: Gemma 3 1B makes their short answers 15 to 32% faster but long prompts 5 to 9% slower, so they use it only after `ts5-tune --models` picks it on your machine.

| Helper | Used for | File size |
| --- | --- | ---: |
| Qwen3 0.6B | Helper (draft) for the Qwen3, Qwen 2.5 Coder and DeepSeek models | 0.4 GB |
| Llama 3.2 1B Instruct | Helper for Llama 3.1 8B and Llama 3.2 3B | 0.8 GB |
| Gemma 3 1B | Helper for Gemma 3 12B and Gemma 4 12B | 0.8 GB |
| Gemma 3 1B (Q8_0) | The same helper at 8 bits, slightly more accurate, a little larger | 1.1 GB |

## Install from a terminal

`ts5 fetch` downloads over HTTPS, continues a stopped download where it left off, and checks the file's SHA-256 before keeping it. Each link below is pinned to the exact upload that was tested.

Run the commands inside the models folder of the way you use TokenStation5:

| You use | Models folder | Run the commands as |
| --- | --- | --- |
| The desktop app, the terminal app, or `ts5` yourself | `~/.local/share/tokenstation/models` | yourself |
| The TV app (`tokenstation-tv`) or the API-only service (`tokenstation`) | `/var/lib/tokenstation/models` | the service's user: put `sudo -u tokenstation` in front of each `ts5 fetch` |

The packages put `ts5` in `/usr/bin`; the portable install puts it in `~/.local/opt/tokenstation/bin`. A 14B model is about 9 GB, so check there's room (`df -h .`).

For the desktop app:

```sh
mkdir -p ~/.local/share/tokenstation/models
cd ~/.local/share/tokenstation/models
```

For the services:

```sh
cd /var/lib/tokenstation/models
# then: sudo -u tokenstation ts5 fetch ...
```

**Qwen3 14B** (9.0 GB, [bartowski/Qwen_Qwen3-14B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-14B-GGUF))

```sh
ts5 fetch https://huggingface.co/bartowski/Qwen_Qwen3-14B-GGUF/resolve/bd080f768a6401c2d5a7fa53a2e50cd8218a9ce2/Qwen_Qwen3-14B-Q4_K_M.gguf Qwen_Qwen3-14B-Q4_K_M.gguf --sha256 915913e22399475dbe6c968ac014d9f1fbe08975e489279aede9d5c7b2c98eb6
```

**Qwen3.5 9B (DeepSeek-V4-Pro heretic)** (5.6 GB, [mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF](https://huggingface.co/mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF))

```sh
ts5 fetch https://huggingface.co/mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF/resolve/3b6f11e221195ee3650d7ac0ffce6f9fe7275d8f/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf --sha256 a692de9520e4c0bb1512deb91e950f38e14854650aff43419112f87e6052c44b
```

**Qwen3.8 9B (distill)** (5.8 GB, [empero-ai/Qwen3.8-9B-Distill-GGUF](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF))

```sh
ts5 fetch https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF/resolve/760121cd70bb4c36b2b5ec58eb765e0df5987efe/Qwen3.8-9B-Q4_K_M.gguf Qwen3.8-9B-Q4_K_M.gguf --sha256 df13d66021cef676f82be74053220fd75af6bf2a6a7fb77f5222ab9e50744a7a
```

**DeepSeek R1 Distill Qwen 14B (abliterated v2)** (9.0 GB, [mradermacher/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2-GGUF))

```sh
ts5 fetch https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2-GGUF/resolve/39e97fd1a97e3e4b5be8ba92d6640a95f2757d29/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2.Q4_K_M.gguf DeepSeek-R1-Distill-Qwen-14B-abliterated-v2.Q4_K_M.gguf --sha256 5bc6f9ce7fbc305c4eeb06253c251e69d9920a5fbac352d73654b63ea8d17e70
```

**Qwen3 8B** (5.0 GB, [bartowski/Qwen_Qwen3-8B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF))

```sh
ts5 fetch https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/0b69f75b7472688e6808490aa2b85efdb81b5ce7/Qwen_Qwen3-8B-Q4_K_M.gguf Qwen_Qwen3-8B-Q4_K_M.gguf --sha256 54fffa050078e984116639c83dfb64b5aa6d4cd474e018b076777c632bbccccd
```

**Llama 3.1 8B Instruct** (4.9 GB, [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF))

```sh
ts5 fetch https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/bf5b95e96dac0462e2a09145ec66cae9a3f12067/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --sha256 7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c
```

**Qwen 2.5 Coder 7B Instruct** (4.7 GB, [bartowski/Qwen2.5-Coder-7B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF))

```sh
ts5 fetch https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/1f629da0c8bed16b9e50cee91c70693650e66c35/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf --sha256 1664fccab734674a50763490a8c6931b70e3f2f8ec10031b54806d30e5f956b6
```

**Qwen 2.5 Coder 14B Instruct** (9.0 GB, [bartowski/Qwen2.5-Coder-14B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF))

```sh
ts5 fetch https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/5b379ec4bf71bafecb5f9081ad28b19939128988/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf --sha256 2946d28c9e1bb2bcae6d42e8678863a31775df6f740315c7d7e6d6b6411f5937
```

**Llama 3.2 3B Instruct** (2.0 GB, [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF))

```sh
ts5 fetch https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-Q4_K_M.gguf Llama-3.2-3B-Instruct-Q4_K_M.gguf --sha256 6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff
```

**Gemma 3 12B** (7.3 GB, [bartowski/google_gemma-3-12b-it-GGUF](https://huggingface.co/bartowski/google_gemma-3-12b-it-GGUF))

```sh
ts5 fetch https://huggingface.co/bartowski/google_gemma-3-12b-it-GGUF/resolve/648e3a36a77c8a9f12d86e741f9dcb9089c769c4/google_gemma-3-12b-it-Q4_K_M.gguf google_gemma-3-12b-it-Q4_K_M.gguf --sha256 fc57f67efa46d711c346e587cbef7d049e95f3df8db2eb2271153343ef0acc7b
```

**Gemma 4 12B** (7.7 GB, [bartowski/gemma-4-12B-it-GGUF](https://huggingface.co/bartowski/gemma-4-12B-it-GGUF))

```sh
ts5 fetch https://huggingface.co/bartowski/gemma-4-12B-it-GGUF/resolve/2ae7d41be21ca62de00a2d320ee9cec50daa3aa6/gemma-4-12B-it-Q4_K_M.gguf gemma-4-12B-it-Q4_K_M.gguf --sha256 3962624dcd25b947d889dc9ae1bf275b61db6cd4dbe694057f34fffef1671509
```

**Ternary Bonsai 2 27B** (7.6 GB, [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf))

```sh
ts5 fetch https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/839bd57023c1558945afea70a74fa83c054f35f9/Ternary-Bonsai-2-27B-Q2_0.gguf Ternary-Bonsai-2-27B-Q2_0.gguf --sha256 4f99aed01b8a877e153f9aa6569a4440fe17c59701cc0953e36d7f460549e70e
```

**Ternary Bonsai 2 9B** (2.6 GB, [prism-ml/Ternary-Bonsai-2-9B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf))

```sh
ts5 fetch https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf/resolve/82265ccc78ddaac4700286a35e9badcc32376dd9/Ternary-Bonsai-2-9B-Q2_0.gguf Ternary-Bonsai-2-9B-Q2_0.gguf --sha256 1f884aa56aacd2bbbb2e4b7abaead2b1102b74a9c75a23fc33dab916813a384d
```

**Ternary Bonsai 2 4B** (1.2 GB, [prism-ml/Ternary-Bonsai-2-4B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf))

```sh
ts5 fetch https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf/resolve/50ebba61be85e5be07e8083db468ace0fc1bac43/Ternary-Bonsai-2-4B-Q2_0.gguf Ternary-Bonsai-2-4B-Q2_0.gguf --sha256 64a5183b44921bdb5ac4a6f5f92007821d6ebe5c66997d3c147fd214ab9433d4
```

### Helpers

**Qwen3 0.6B** (0.4 GB, [unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF))

```sh
ts5 fetch https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_K_M.gguf Qwen3-0.6B-Q4_K_M.gguf --sha256 ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a
```

**Llama 3.2 1B Instruct** (0.8 GB, [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF))

```sh
ts5 fetch https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/067b946cf014b7c697f3654f621d577a3e3afd1c/Llama-3.2-1B-Instruct-Q4_K_M.gguf Llama-3.2-1B-Instruct-Q4_K_M.gguf --sha256 6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83
```

**Gemma 3 1B** (0.8 GB, [bartowski/google_gemma-3-1b-it-GGUF](https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF))

```sh
ts5 fetch https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/116f76234503685a98f572982177b11d44ec8ff1/google_gemma-3-1b-it-Q4_K_M.gguf google_gemma-3-1b-it-Q4_K_M.gguf --sha256 12bf0fff8815d5f73a3c9b586bd8fee8e7b248c935de70dec367679873d0f29d
```

**Gemma 3 1B (Q8_0)** (1.1 GB, [bartowski/google_gemma-3-1b-it-GGUF](https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF))

```sh
ts5 fetch https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/116f76234503685a98f572982177b11d44ec8ff1/google_gemma-3-1b-it-Q8_0.gguf google_gemma-3-1b-it-Q8_0.gguf --sha256 375e12a4a18929a641f9744b060d4a7cf4e279530750555828ec0c117870bc96
```

## After downloading

- The app and the terminal app list a new model within a few seconds. The API-only service uses `MODEL` in `/etc/default/tokenstation` (or the first file in its folder); restart it after a change.
- These models need no tuning: their settings are built in, and each uses its helper when the helper's file is in the same folder. `ts5-tune --models` (or **T** on the terminal app's Models page) is for other models, or to measure again on your own machine.
- Each model is under its own license: Llama and Gemma have terms of use of their own. The model files are not part of TokenStation5.
