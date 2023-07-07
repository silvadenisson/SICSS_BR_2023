# Traducao codigo em python para R - workshop Modern NLP SICSS 2023
# Denisson Silva #
# ambiente windows / visual studio community 2022

# Autor: Lucas Roberto
# codigo origial: https://colab.research.google.com/drive/1Kr64VrhouJ79kZwN7yXpM8zXMmTuSQ3g?usp=sharing

# Referências
# Curso Transformers -> https://huggingface.co/course/chapter1/1
# Curso DeepLearning -> https://www.youtube.com/watch?v=_QUEXsHfsA0&list=PLfYUBJiXbdtRL3FMB3GoWHRI8ieU6FhfM
# Curso Deep NLP Stanford -> https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ
# Curso Deep Learning -> https://www.youtube.com/watch?v=zQY2YvkMbHI&list=PLkRLdi-c79HKEWoi4oryj-Cx-e47y_NcM

# Bibliotecas
library(reticulate)

# ambiente e bliotecas python
# conda_create("r-reticulate")
# conda_install("r-reticulate", "transformers")
# conda_install("r-reticulate", "PyTorch")

use_condaenv("r-reticulate")

# importando transformers
transformers <- reticulate::import("transformers")

# Instantiate a pipeline
classifier <- transformers$pipeline(task = "sentiment-analysis")

texto <- "We are very happy to show you the 🤗 Transformers library."

output <- classifier(texto)
output

textos2 <- list("We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it.")

results <- classifier(textos2)

tweets <- list('i hate when u put something like cake or a drink into the fridge n then suddenly it TASTES like the fridge',
              'I hate people who act so big on loyalty but do snake shit.', 
              'i hate random sad days. having a heavy heart and an anxious mind is the worst.')


results <- classifier(tweets)

tweets <- list("Twitter's board approves @elonmusk's bid of $44 billion in a unanimous decision. Leftist minds are definitely going to explode.",
          'I will keep supporting Dogecoin',
          'Happy Father’s Day')

results <- classifier(tweets)

# utilizando outros modelos

# dando error
model_path <- "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task <- transformers$pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

tweets  <-list('Aneel anuncia reajuste de até 64% em valor de taxa extra da conta de luz', 
               'Uma caixa de bom bom com o Prestígio da Nestlé, Caribe do Garoto e Amandita da Lacta seria a perfeição.', 
               'Vasco da grana pra sempre... vou te amar... e na barreira eu vou festejar... e cantar... outra vez... com os loucos da saída 3🎶')


results <- sentiment_task(tweets)

# discurso de odio

df <- read.csv('https://raw.githubusercontent.com/JAugusto97/ToLD-Br/main/ToLD-BR.csv')

df |> head()

tweets <- as.list(df$text[1:10]) # selecionei 10 primeiros para ser mais rapido o teste

model_path <- 'Hate-speech-CNERG/dehatebert-mono-portugese'
hs_model <- transformers$pipeline('text-classification', model=model_path, tokenizer=model_path)

result <- hs_model(tweets)

# ZeroShot classification

nlp <- transformers$pipeline("zero-shot-classification", model="joeddav/bart-large-mnli-yahoo-answers")

sequence_to_classify <- "Who are you voting for in 2020?"
candidate_labels <- list("Europe", "public health", "politics", "elections")
hypothesis_template <- "This text is about {}."

nlp(sequence_to_classify, candidate_labels, multi_label=T)

