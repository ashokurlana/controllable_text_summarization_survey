# Controllable Text Summarization Survey
\
This repository contains the controllable text summarization (CTS) survey papers and is based on our paper, ["Controllable Text Summarization: Unraveling Challenges, Approaches, and Prospects -- A Survey"](https://arxiv.org/abs/2311.09212)

You can cite our paper as the following
```
@misc{urlana2023controllable,
      title={Controllable Text Summarization: Unraveling Challenges, Approaches, and Prospects -- A Survey}, 
      author={Ashok Urlana and Pruthwik Mishra and Tathagato Roy and Rahul Mishra},
      year={2023},
      eprint={2311.09212},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
We group the papers according to the controllable aspects as [Length](#length), [Coverage](#coverage), [Style](#style), [Abstractivity](#abstractivity), [Salience](#salience),  [Entity](#entity), [Topic](#topic), [Role](#role), [Diversity](#diversity), [Structure](#structure).

# Length
| Paper | Datasets Used| 
| -- | --- |
|MACSUM: Controllable Summarization with Mixed Attributes [TACL -2023](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00575/116714) [code](https://github.com/yizhuliu/lengthcontrol) [data](https://huggingface.co/datasets/chatc/MACSum)|CNN Daily Mail, QMSum|
|Abstractive Document Summarization with Summary-length Prediction [EACL-2023](https://aclanthology.org/2023.findings-eacl.45)|CNNDM, NYT, WikiHow|
|Length Control in Abstractive Summarization by Pretraining Information Selection [ACL-2022](https://aclanthology.org/2022.acl-long.474) [code](https://github.com/yizhuliu/lengthcontrol)|CNN-DailyMail, XSUM|
|Generating Multiple-Length Summaries via Reinforcement Learning for Unsupervised Sentence Summarization [EMNLP-2022](https://aclanthology.org/2022.findings-emnlp.214) [code](https://github.com/dmhyun/MSRP)|DUC2004|
|A Character-Level Length-Control Algorithm for Non-Autoregressive Sentence Summarization [Neurips-2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/bb0f9af6a4881ccb6e14c11b8b4be710-Abstract-Conference.html) [code](https://github.com/MANGA-UOFA/NACC)|Gigaword, DUC2004|
|CTRLSUM: Towards Generic Controllable Text Summarization [EMNLP-2022](https://aclanthology.org/2022.emnlp-main.396) [code](https://github.com/salesforce/ctrl-sum)|CNNDM, arXiv, BIGPATENT|
|A New Approach to Overgenerating and Scoring Abstractive Summaries [NAACL-2021](https://aclanthology.org/2021.naacl-main.110) [code](https://github.com/ucfnlp/varying-length-summ) [data](https://github.com/ucfnlp/varying-length-summ/tree/main/data)|Gigaword, Newsroom|
|Controllable Summarization with Constrained Markov Decision Process [TACL-2021](https://aclanthology.org/2021.tacl-1.72) [code](https://github.com/kenchan0226/control-sum-cmdp)|CNNDM, Newsroom, DUC-2002|
|Lenatten: An effective length controlling unit for text summarization [ACL-2021](https://aclanthology.org/2021.findings-acl.31) [code](https://github.com/X-AISIG/LenAtten)|CNNDM|
|Interpretable multi headed attention for abstractive summarization at controllable lengths [COLING-2020](https://aclanthology.org/2020.coling-main.606)|MSR Narratives and Thinking-Machines|
|Positional Encoding to Control Output Sequence Length [NAACL-2019](https://aclanthology.org/N19-1401) [code](https://github.com/takase/control-length)|JAMUS corpus (Japanese) of different number of characters present in the summary|
|Global Optimization under Length Constraint for Neural Text Summarization [ACL-2019](https://aclanthology.org/P19-1099)|CNNDM, Mainichi|
|A Large-Scale Multi-Length Headline Corpus for Analyzing Length-Constrained Headline Generation Model Evaluation [INLG-2019](https://aclanthology.org/W19-8641) [data](https://cl.asahi.com/api_data/jnc-jamul-en.html)|JAMUS corpus (Japanese) of different number of characters present in the summary|
|Controllable Abstractive Summarization [ACL-NMT(W)-2018](https://aclanthology.org/W18-2706)|CNN-DailyMail |
|Unsupervised Sentence Compression using Denoising Auto-Encoders [CoNLL-2018](https://aclanthology.org/K18-1040) [code](https://github.com/zphang/usc_dae)|Gigaword|
|Controlling Length in Abstractive Summarization Using a Convolutional Neural Network [EMNLP-2018](https://aclanthology.org/D18-1444) [code](https://github.com/YizhuLiu/sumlen)|CNNDM, DMQA|
|Controlling Output Length in Neural Encoder-Decoders [EMNLP-2016](https://aclanthology.org/D16-1140) [code](https://github.com/kiyukuta/lencon)|DUC2004, Gigaword|
|A Neural Attention Model for Abstractive Sentence Summarization [EMNLP-2015](https://aclanthology.org/D15-1044)|NYT, DUC2004|
# Coverage
| Paper | Datasets Used | 
| -- | --- |
|MACSUM: Controllable Summarization with Mixed Attributes [TACL -2023](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00575/116714) [code](https://github.com/yizhuliu/lengthcontrol) [data](https://huggingface.co/datasets/chatc/MACSum)|CNN Daily Mail, QMSum|
|SWING : Balancing Coverage and Faithfulness for Dialogue Summarization [EACL-2023](https://aclanthology.org/2023.findings-eacl.37) [code](https://github.com/amazon-science/AWS-SWING)|DIALOG-SUM, SAMSUM|
|Unsupervised Multi-Granularity Summarization [EMNLP-2022](https://aclanthology.org/2022.findings-emnlp.366) [data](https://github.com/maszhongming/GranuDUC)|GranuDUC, MultiNews, DUC2004, Arxiv|
|Multi-LexSum: Real-World Summaries of Civil Rights Lawsuits at Multiple Granularities [NIPS-2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/552ef803bef9368c29e53c167de34b55-Abstract-Datasets_and_Benchmarks.html) [code](https://github.com/multilexsum/dataset) [data](https://multilexsum.github.io)|Multi-LexSum|
|Controllable Abstractive Dialogue Summarization with Sketch Supervision [ALC-IJCNLP-2021](https://aclanthology.org/2021.findings-acl.454) [code](https://github.com/salesforce/ConvSumm)|SAMSum|
|SemSUM: Semantic Dependency Guided Neural Abstractive Summarization [AAAI-2020](https://ojs.aaai.org/index.php/AAAI/article/view/6312) [data](https://github.com/harvardnlp/sent-summary (one link is not accessible))|Gigaword, DUC2004 and MSR abstractive summarization dataset|
|Get to the point: Summarization with pointer generator networks [ACL-2017](https://aclanthology.org/P17-1099) [code](https://github.com/abisee/pointer-generator)|CNNDM|
# Style
| Paper | Datasets Used | 
| -- | --- |
|Overview of the BioLaySumm 2023 Shared Task on Lay Summarization of Biomedical Research Articles [ACL-BIoNLP(W)-2023](https://aclanthology.org/2023.bionlp-1.44)|PLOS and eLife|
|Generating Summaries with Controllable Readability Levels [EMNLP-2023](https://arxiv.org/abs/2310.10623) [code](https://github.com/amazon-science/controllable-readability-summarization)|CNNDM|
|HYDRASUM: Disentangling Style Features in Text Summarization with Multi-Decoder Models [EMNLP-2022](https://aclanthology.org/2022.emnlp-main.30) [code](https://github.com/salesforce/hydra-sum)|CNN Daily Mail, XSUM, Newsroom|
|Readability Controllable Biomedical Document Summarization [EMNLP-2022](https://aclanthology.org/2022.findings-emnlp.343) [data](http://www.nactem.ac.uk/readability/)|TS and PLS|
|Inference time style control for summarization  [NAACL-2021](https://aclanthology.org/2021.naacl-main.476) [code](https://github.com/ShuyangCao/inference_style_control)|CNNDM|
|Hooks in the Headline: Learning to Generate Headlines with Controlled Styles [ACL-2020](https://aclanthology.org/2020.acl-main.456) [code](https://github.com/jind11/TitleStylist)|NYT, CNN|
|Generating Formality-tuned Summaries Using Input-dependent Rewards [CoNLL-2019](https://aclanthology.org/K19-1078)|CNN Daily Mail + Webis-TLDR-17 corpus|
|Controllable Abstractive Summarization [ACL-NMT(W)-2018](https://aclanthology.org/W18-2706)|CNN-DailyMail |
# Abstractivity
| Paper | Datasets Used | 
| -- | --- |
|Controllable Summarization with Constrained Markov Decision Process [TACL-2021](https://aclanthology.org/2021.tacl-1.72) [code](https://github.com/kenchan0226/control-sum-cmdp)|CNNDM, Newsroom, DUC-2002|
|Controlling the Amount of Verbatim Copying in Abstractive Summarization [AAAI-2020](https://ojs.aaai.org/index.php/AAAI/article/view/6420) [code](https://github.com/ucfnlp/control-over-copying)|Gigaword, Newsroom|
|Improving Abstraction in Text Summarization [EMNLP-2018](https://aclanthology.org/D18-1207)|CNNDM|
|Get to the point: Summarization with pointer generator networks [ACL-2017](https://aclanthology.org/P17-1099) [code](https://github.com/abisee/pointer-generator)|CNNDM|
|SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents [AAAI-2017](https://ojs.aaai.org/index.php/AAAI/article/view/10958) [code](https://github.com/hpzhao/SummaRuNNer)|CNN/DM, DUC2002|
# Salience
| Paper | Datasets Used | 
| -- | --- |
|Incorporating Question Answering-Based Signals into Abstractive Summarization via Salient Span Selection [EACL-2023](https://aclanthology.org/2023.eacl-main.42)|CNNDM, XSUM, NYTimes|
|SOCRATIC Pretraining: Question-Driven Pretraining for Controllable Summarization [ACL-2023](https://aclanthology.org/2023.acl-long.713) [code](https://github.com/salesforce/socratic-pretraining)|QMSum and SQuALITY|
|Guiding Generation for Abstractive Text Summarization based on Key Information Guide Network [NAACL-HLT-2018](https://aclanthology.org/N18-2009)|CNNDM|
|SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents [AAAI-2017](https://ojs.aaai.org/index.php/AAAI/article/view/10958) [code](https://github.com/hpzhao/SummaRuNNer)|CNN/DM, DUC2002|
# Entity
| Paper | Datasets Used | 
| -- | --- |
|SOCRATIC Pretraining: Question-Driven Pretraining for Controllable Summarization [ACL-2023](https://aclanthology.org/2023.acl-long.713) [code](https://github.com/salesforce/socratic-pretraining)|QMSum and SQuALITY|
|Extractive Entity-Centric Summarization as Sentence Selection using Bi-Encoders [AACL-2022](https://aclanthology.org/2022.aacl-short.40)|EntSum|
|CTRLSUM: Towards Generic Controllable Text Summarization [EMNLP-2022](https://aclanthology.org/2022.emnlp-main.396) [code](https://github.com/salesforce/ctrl-sum)|CNNDM, arXiv, BIGPATENT|
|ENTSUM: A Data Set for Entity-Centric Summarization [ACL-2022](https://aclanthology.org/2022.acl-long.237) [code](https://github.com/bloomberg/entsum) [data](https://github.com/bloomberg/entsum)|CNNDM, NYT|
|Controllable Summarization with Constrained Markov Decision Process [TACL-2021](https://aclanthology.org/2021.tacl-1.72) [code](https://github.com/kenchan0226/control-sum-cmdp)|CNNDM, Newsroom, DUC-2002|
|Controllable Neural Dialogue Summarization with Personal Named Entity Planning [EMNLP-2021](https://api.semanticscholar.org/CorpusID:237941123) [code](https://github.com/seq-to-mind/planning_dial_summ)|SAMSum|
|Controllable Abstractive Sentence Summarization with Guiding Entities [COLING-2020](https://aclanthology.org/2020.coling-main.497) [code](https://github.com/thecharm/Abs-LRModel/tree/main)|Gigaword, DUC2004|
|Controllable Abstractive Summarization [ACL-NMT(W)-2018](https://aclanthology.org/W18-2706)|CNN-DailyMail |
# Topic
| Paper | Datasets Used | 
| -- | --- |
|MACSUM: Controllable Summarization with Mixed Attributes [TACL -2023](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00575/116714) [code](https://github.com/yizhuliu/lengthcontrol) [data](https://huggingface.co/datasets/chatc/MACSum)|CNN Daily Mail, QMSum|
|Topic-aware Multimodal Summarization [AACL-2022](https://aclanthology.org/2022.findings-aacl.36) [code](https://github.com/mailsourajit25/Topic-Aware-Multimodal-Summarization) [data](https://drive.google.com/file/d/1yzEE_n5q2VaqMrI1q9BrTS9ebrF-nQko/view?usp=sharing)|MSMO|
|NEWTS: A Corpus for News Topic-Focused Summarization [ACL-2022](https://aclanthology.org/2022.findings-acl.42) [data](https://github.com/ali-bahrainian/NEWTS)|NEWTS|
|ASPECTNEWS: Aspect-Oriented Summarization of News Documents [ACL-2022](https://aclanthology.org/2022.acl-long.449) [code](https://github.com/tanyuqian/aspect-based-summarization) [data](https://github.com/oja/aosumm/tree/master/data)|ASPECTNEWS|
|Aspect-controllable opinion summarization [EMNLP-2021](https://aclanthology.org/2021.emnlp-main.528) [code](https://github.com/rktamplayo/AceSum)|SPACE, OPOSUM+|
|Decision-Focused Summarization [EMNLP-2021](https://aclanthology.org/2021.emnlp-main.10) [code](https://github.com/ChicagoHAI/decsum) [data](https://github.com/ChicagoHAI/decsum)|Yelp's businesses, reviews, and user data|
|CATS: Customizable Abstractive Topic-based Summarization [ACM-2021](https://dl.acm.org/doi/10.1145/3464299) [code](https://github.com/ali-bahrainian/CATS)|CNNDM, AMI , ICSI, ADSE|
|WikiAsp: A Dataset for Multi-domain Aspect-based Summarization [TACL-2021](https://aclanthology.org/2021.tacl-1.13) [code](https://github.com/neulab/wikiasp) [data](https://huggingface.co/datasets/wiki_asp)|WikiAsp|
|Summarizing Text on Any Aspects: A Knowledge-Informed Weakly-Supervised Approach [EMNLP-2020](https://aclanthology.org/2020.emnlp-main.510) [code](https://github.com/tanyuqian/aspect-based-summarization)|CNN -Dailymail, MA News, All the News|
|OPINIONDIGEST: A Simple Framework for Opinion Summarization [ACL-2020](https://aclanthology.org/2020.acl-main.513) [code](https://github.com/megagonlabs/opiniondigest)|Hotel, Yelp|
|Read what you need: Controllable Aspect-based Opinion Summarization of Tourist Reviews [SIGIR-2020](https://dl.acm.org/doi/10.1145/3397271.3401269) [code](https://github.com/rajdeep345/ControllableSumm) [data](https://github.com/rajdeep345/ControllableSumm)|Tourism Reviews|
|Generating topic-oriented summaries using neural attention [NAACL-HLT-2018](https://aclanthology.org/N18-1153)|CNNDM|
|Vocabulary Tailored Summary Generation [ACL-2018](https://aclanthology.org/C18-1068)|CNNDM|
# Role
| Paper | Datasets Used | 
| -- | --- |
|Other Roles Matter! Enhancing Role-Oriented Dialogue Summarization via Role Interactions [ACL-2022](https://aclanthology.org/2022.acl-long.182) [code](https://github.com/xiaolinAndy/RODS) [data](https://github.com/xiaolinAndy/RODS/tree/main/data/MC)|CSDS, MC|
|Towards Modeling Role-Aware Centrality for Dialogue Summarization [AACL-2022](https://aclanthology.org/2022.aacl-short.6) [data](https://github.com/xiaolinAndy/RODS/tree/main/data/MC)|CSDS, MC|
|CSDS: A fine-grained Chinese dataset for customer service dialogue summarization [EMNLP-2021](https://aclanthology.org/2021.emnlp-main.365) [code](https://github.com/xiaolinAndy/CSDS) [data](https://github.com/Shen-Chenhui/MReD)|CSDS|
# Diversity
| Paper | Datasets Used | 
| -- | --- |
|A Well-Composed Text is Half Done! Composition Sampling for Diverse Conditional Generation [ACL-2022](https://aclanthology.org/2022.acl-long.94) [code](https://github.com/google-research/language/tree/master/language/frost)|CNN/DailyMail and Xsum and question generation (SQuAD)|
# Structure
| Paper | Datasets Used | 
| -- | --- |
|STRONG – Structure Controllable Legal Opinion Summary Generation [IJCNLP-AACL-2023](https://arxiv.org/abs/2309.17280)|CanLII|
|SentBS: Sentence-level beam search for controllable summarization [EMNLP-2022](https://aclanthology.org/2022.emnlp-main.699) [code](https://github.com/Shen-Chenhui/SentBS)|Meta Review Dataset (MReD)|
|MReD: A Meta-Review Dataset for Structure-Controllable Text Generation [ACL-2022](https://aclanthology.org/2022.findings-acl.198) [code](https://github.com/Shen-Chenhui/MReD) [data](https://github.com/Shen-Chenhui/MReD)|MReD|
|Planning with Learned Entity Prompts for Abstractive Summarization [TACL-2021](https://aclanthology.org/2021.tacl-1.88)|CNN/DailyMail, XSum, SAMSum, and BillSum|


