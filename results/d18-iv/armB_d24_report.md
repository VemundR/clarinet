# nanochat training report

Generated: 2026-06-10 13:41:36

## Environment

### Git Information
- Branch: clarinet/results-d18-iv
- Commit: 926b3f4 (dirty)
- Message: Drop duplicate root iv.log (kept in results/d18-iv/)

### Hardware
- Platform: Linux
- CPUs: 64 cores (128 logical)
- Memory: 1007.4 GB
- GPUs: 1x NVIDIA H100 80GB HBM3
- GPU Memory: 79.2 GB total
- CUDA Version: 12.8
- Hourly Rate: $3.00/hour

### Software
- Python: 3.10.20
- PyTorch: 2.9.1+cu128


### Bloat
- Characters: 674,010
- Lines: 15,057
- Files: 69
- Tokens (approx): 168,502
- Dependencies (uv.lock lines): 3,405

Run started: 2026-06-10 13:41:36

---

## Tokenizer training
timestamp: 2026-06-10 13:43:01

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 32,768
- train_time: 79.8106
- num_special_tokens: 12
- token_bytes_min: 1
- token_bytes_max: 32
- token_bytes_mean: 6.5826
- token_bytes_std: 2.8124


## Tokenizer evaluation
timestamp: 2026-06-10 13:43:12

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 405 | 4.49 | -0.2% |
| korean | 893 | 745 | 1.20 | 749 | 1.19 | -0.5% |
| code | 1259 | 576 | 2.19 | 397 | 3.17 | +31.1% |
| math | 1834 | 936 | 1.96 | 911 | 2.01 | +2.7% |
| science | 1112 | 260 | 4.28 | 247 | 4.50 | +5.0% |
| fwe-train | 2948778 | 631304 | 4.67 | 622492 | 4.74 | +1.4% |
| fwe-val | 3024593 | 653067 | 4.63 | 644915 | 4.69 | +1.2% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 405 | 4.49 | -4.7% |
| korean | 893 | 364 | 2.45 | 749 | 1.19 | -105.8% |
| code | 1259 | 309 | 4.07 | 397 | 3.17 | -28.5% |
| math | 1834 | 832 | 2.20 | 911 | 2.01 | -9.5% |
| science | 1112 | 249 | 4.47 | 247 | 4.50 | +0.8% |
| fwe-train | 2948778 | 611619 | 4.82 | 622492 | 4.74 | -1.8% |
| fwe-val | 3024593 | 631183 | 4.79 | 644915 | 4.69 | -2.2% |


## Base model training
timestamp: 2026-06-11 11:18:42

- run: clarinet-ab-iv
- device_type: 
- fp8: True
- fp8_recipe: tensorwise
- depth: 18
- aspect_ratio: 64
- head_dim: 128
- max_seq_len: 2048
- window_pattern: SSSL
- num_iterations: -1
- target_flops: -1.0000
- target_param_data_ratio: 8.0000
- device_batch_size: 16
- total_batch_size: -1
- embedding_lr: 0.3000
- unembedding_lr: 0.0080
- weight_decay: 0.2800
- matrix_lr: 0.0200
- scalar_lr: 0.5000
- warmup_steps: 40
- warmdown_ratio: 0.6500
- final_lr_frac: 0.0500
- resume_from_step: -1
- eval_every: 250
- eval_tokens: 41,943,040
- core_metric_every: 2000
- core_metric_max_per_task: 500
- sample_every: 2000
- save_every: -1
- model_tag: d18-iv
- Number of parameters: 701,891,594
- Number of FLOPs per token: 2.179995e+09
- Calculated number of iterations: 2475
- Number of training tokens: 2,595,225,600
- Tokens : Scaling params ratio: 8.0000
- DDP world size: 1
- warmup_steps: 40
- warmdown_ratio: 0.6500
- final_lr_frac: 0.0500
- Minimum validation bpb: 0.8467
- Final validation bpb: 0.8467
- CORE metric estimate: 0.1401
- MFU %: 50.01%
- Total training flops: 5.657580e+18
- Total training time: 192.43m
- Peak memory usage: 32421.59MiB


## Base model evaluation
timestamp: 2026-06-11 11:58:30

- model: base_model (step 2475)
- CORE metric: 0.1376
- train bpb: 0.8624
- val bpb: 0.8617
- hellaswag_zeroshot: 0.1119
- jeopardy: 0.0090
- bigbench_qa_wikidata: 0.2623
- arc_easy: 0.4265
- arc_challenge: 0.0796
- copa: -0.0400
- commonsense_qa: 0.0653
- piqa: 0.3069
- openbook_qa: 0.0240
- lambada_openai: 0.1609
- hellaswag: 0.1472
- winograd: 0.0916
- winogrande: -0.0024
- bigbench_dyck_languages: 0.0850
- agi_eval_lsat_ar: 0.0272
- bigbench_cs_algorithms: 0.4364
- bigbench_operators: 0.2476
- bigbench_repeat_copy_logic: 0.0625
- squad: 0.3258
- coqa: 0.1953
- boolq: -0.1693
- bigbench_language_identification: 0.1740
- sample 0: <|bos|>The capital of France is the 1. The 1. The 2. The 3.
- sample 1: <|bos|>The chemical symbol of gold is a symbol of the element gold. The symbol of gold is a symbol of the
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Friday. The day of the week is Friday, and the day of the week
- sample 3: <|bos|>The opposite of hot is a 1. The 1. The 2. The 3.
- sample 4: <|bos|>The planets of the solar system are: The solar system is a system of planets, stars, and planets. The solar
- sample 5: <|bos|>My favorite color is a 1. 1. 1. 1. 1.
- sample 6: <|bos|>If 5*x + 3 = 13, then x is 1. 1. 1. 2. 2. 
- unconditioned 0: <|bos|>IfASOur IncreaseMain menuPlant Page ReferenceRandom A BearSqueaky Seed 3Bomb Pockets and A Tetra Three Game-like powers and run number holds 2 clear foam Dragoons with r Intelligence244 StarSqueaky Seed Xavier 3combTYSSSpecifically, we like a variety of species of wind-powered ships. The red and black in the image indicates the box I believe it is originally from, but I will admit I haven't checked extensively for specifics of mine out, but with a little imagination and random chance, it's a lot easier to see a starting point.
Except
- unconditioned 1: <|bos|>AlFindInter, CHUalaeru, aka SatOn"We put the-wharm.
Sometacz Philipp Siberian youth chess, Bonnic’s team benefits from year five, who traded as Julie Carb Alekose during the 20 king-of-meters tradition. The older players in the Championship played faster and had so much confidence compared to a century ago.

They could make very big moves depending on their squad and know for a deity or some other reason. The flexible equation could have been established through consistency in the goal design of obscurity real teamWesters took the country to another level as players opted for steady
- unconditioned 2: <|bos|>YourGM81chis'idmustainableest ( Higher Day Of Justice And Enrica Being Diplanned, An Employment that Is Deemd a Rendoku Federation More Recommended legislation enhanced, and both mundane and providence upon twenty-eight most important areas.is hygral contouring of universities hold two scripted competent causes by the same long-term fabill to do the proper essay benefits options university of physics and technology When I meddisc purposes(auniverse) play a key essay for a virtual human getting into IDEXSistors fungi of the sea underscores the critical contemporary field of film and communications from its former
- unconditioned 3: <|bos|>Model,
On 0, what ratio focused targeting in a first stage ice coal-fired powerplant, timing involved grave problems for international trade in powerlifting with facilities in exploiting human soldiers. For more, everybody should learn to do it, fixed benefits for HR on paris script are clearly unnecessary, which means that nothing that could enhance: presumably training and this 30 Agree4 most other equipment, management practices [for planning, slightly assimilation, evaluation], performance medical students, and behavioral 3 has therefore, a reported positivity. Cooperation locations in the ring of time significance if physical, binary fatigue failure emergency.
The field of crew
- unconditioned 4: <|bos|>#CSection III squareFind eros you (OR ex to confery .......surprisingly been engrossing in greeting the world generations forward. When we go to be a Christian product, there are craftsmanship and one other material qualities we end some time something. Painting and winterizing or 'home tactics and games' has no meat-land if they are excellent and inflammable.Dine-by Debora chauierbakingtechnologygurl1 through pancop wet somehow succeeds in your ancestor x that new, meaning it's not going to shoot some places invariably into caring by moldings turning here for as distinct as
- unconditioned 5: <|bos|>YouAre RosenE1W. A Wild pumpkin to make an effect measuring articulateby0 certain colours efficiently emitted from their side of the corner, and a thing is essential to estate for1W the strength string advantages unusable in basic shapes, the best promoting an individual of50 vigence 198; reason that these character locations are three Wolfenscent.Cf. Double radius equilateral arches and the individuals being spotlight converinter? Figure 3. 1 Third Clinovasu7f business maven willing casting upon  following axis which communicated by: Did desired reseting day amounts, dating's biblical speekn
- unconditioned 6: <|bos|>How chfL y  Cor IGS Cornsys used 2 (20 minss 1 The additional costs and 7.8K 850 MHz Worth 3 Pendies management. FFU. The measures 21848 degrees to the Air extracted daily diagram explained Have what yyu means, ninety consists of some things that were used in classic activated stone materials used to enhance hypersensitivity spectrum Labs made use of ae techq, flyup stabilization equipment, and use of standard •'s printed 1990 Buyers' Primary Uses five of the Gl in Euclid's study
- unconditioned 7: <|bos|>$\MyhomeFun<wi23 down |Hu Qucenter in a problem.How Very Difficult Is It? Sometimes, a dog shouldn't be thrown down to the unthrifty overwhelming crowd. The dog has a functioning mouth, which has a very fine contour and anterior 13 and storks of conductivity yama long a man dogmanippet.

1. Freshwater Bamboo question, what is it?

Registered

Go here. Do you know what a dog sled ride is known?

g/Shutterstock.com/Fe_Yoshi

2. How Important is the Saw Essay how?

Even if Cavalier


## Chat evaluation sft
timestamp: 2026-06-11 07:37:05

- source: sft
- task_name: None
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: d24-base
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.5160
- ARC-Challenge: 0.4104
- MMLU: 0.3363
- GSM8K: 0.1804
- HumanEval: 0.1890
- SpellingBee: 0.9922
- ChatCORE metric: 0.3409


## Summary

- Characters: 674,010
- Lines: 15,057
- Files: 69
- Tokens (approx): 168,502
- Dependencies (uv.lock lines): 3,405

| Metric          | BASE     | SFT      | RL       |
|-----------------|----------|----------|----------|
| CORE            | 0.1376   | -        | -        |
| ARC-Challenge   | -        | 0.4104   | -        |
| ARC-Easy        | -        | 0.5160   | -        |
| GSM8K           | -        | 0.1804   | -        |
| HumanEval       | -        | 0.1890   | -        |
| MMLU            | -        | 0.3363   | -        |
| ChatCORE        | -        | 0.3409   | -        |

Total wall clock time: 17h55m
