# Third-party notices

NaiveDiffusion embeds tokenizer data from other projects. Their licenses are listed here.

## CLIP tokenizer

`src/NaiveDiffusion/Assets/ClipTokenizer` is the BPE vocabulary and merge table of [OpenAI CLIP](https://github.com/openai/CLIP), under the MIT license.

```
Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Qwen tokenizer

`src/NaiveDiffusion/Assets/Qwen2Tokenizer` is the vocabulary and merge table of the [Qwen](https://github.com/QwenLM/Qwen3) tokenizer by Alibaba Cloud, under the Apache License 2.0. The system prompt in `QwenImagePrompt.cs` comes from [Qwen-Image](https://github.com/QwenLM/Qwen-Image), under the same license.

## T5 tokenizer

`src/NaiveDiffusion/Assets/T5Tokenizer` is the tokenizer of [T5](https://github.com/google-research/text-to-text-transfer-transformer) by Google, under the Apache License 2.0.

The text of the Apache License 2.0 is in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
