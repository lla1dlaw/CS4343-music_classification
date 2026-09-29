


#let project(title: "", authors: (), date: none, body) = {
  set document(title: title, author: authors, date: auto)

  align(center + horizon)[
    #text(17pt, weight: "bold", title)
    
    #v(1em) // Vertical spacing
    #authors.join(", ")
    
    #v(1em)
    #date
  ]
  
  v(2em) // Space before the main text begins
  

  body
}

#show: project.with(
  title: "A Comparison of CNN Architectures for Music Genre Classification",
  authors: (
    "Joseph Tully",
    "Liam Laidlaw" 
    ,
  ),
  date: datetime.today().display(),
)
#pagebreak()

// --- Table of Contents ---
#outline()
#pagebreak()




#set heading(numbering: "1.1 |")
= Problem and Motivation
== Purpose

The purpose of this project is to attempt to develop a model using the same architecture as ResNet@ResNet or AlexNet@AlexNet that uses less parameters than their predecessors and that can obtain performance on par with or exeeding these models. We plan to test 4 models, the smallest published versions of the official ResNet and AlexNet models, and two custom models with the same architecture and less parameters. The performance of these models will be compared and analyzed in an attempt to explain differences in performance between the different architectures and across the different parameter counts for each model. The models will then be ranked based on performance.  

== Why Does it Matter?

Categorizing music. Our model will be able to take in music and determine what genre the music is. We will compare our model to the Cyanite AI model. This model is widely accepted as one of the most accurate, hitting marks as close as 95%. Our goal is to meet or beat this score. People around the world use apps like shazam, which can identify exact songs. Where our model comes in handy is for apps such as spotify or apple music. These applications have genre specific playlists and screens within the application. Our model could be used as one of these 'ease of life' features. 

== Research Questions / Hypothesis

We would like to find out just what kind of feature patterns the model will find when looking at the spectrograph. But other research questions arose as a result. What would be a good training set size for the model? How many different patterns can direct the model to a specific genre? 

We do have our initial thoughts though. We believe the model will search for specific locations on the mels spectrograph, and if the graph is particularly active in those areas, its a specific genre. As for training set we found roughly 20 thousand files. And are going to start with a small amount of these files for training somewhere between 5,000 and 10,000.

= Proposed Solution & Course connection
== Connection To Course: CS-4343

CS 4343 introduces the concept of Convolutional Neural Networks (CNNs) by analyzing the architecture of groundbreaking computer vision models including AlexNet@AlexNet and ResNet@ResNet. AlexNet most notably won the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC), and ResNet who famously introduced the idea of skip connections in side of convolutional layers which decreased the effects of the vanishing gradient problem in extremely deep CNNs@ResNet. Our group has decided to benchmark our model's performance against these models using the metrics listed in section 4.

= Data Plan 

Our data set is from Kaggle@data_set and includes 18,829 128x431 pixel  Mel-Spectrograms, each representing 10 seconds of a song with an accompanying genre label. The spectrograms are stored as serialized numpy arrays totalling 4.16GB in size, and will be loaded into custom data loader objects for training and testing. Our model will use a convolutional network to learn the features on its own. We will use 70% of the data set for training (13,180 samples), 10% for validation (1,883 samples), and 20% for testing (3,765). The dataset used in this project is publicly available, free to use, and openly available through Kaggle@data_set.

= Model Evaluation and Analysis
To compare the models, our group will use various different classification performance metrics including Mean Accuracy & Variance, Precision, Recall, and F1-Score. We will also include analysis of confusion matrices for the various models across the genres included in the testing data. These metrics will be compiled into a final comparative performance analysis of each model. Successful completion of this experiment would yield strong findings derived from this analysis that provide insight into the benefits and drawbacks of various CNN architectures when used for mel-spectrogram music genre classification. 

= Feasibility, resources, and risks
== Software Frameworks, and Potential Dependencies

We will be using Vscode as our compiler, and Github as our cloud based storage / share software. We will be writing this project in python.
=== Potential Dependencies
- pytorch
- numpy
- kaggle hub

== Technical / Data Risks

By using an open source data set that is provided from unknown sources, we have some potential risks with our data. This includes if some genre's overlap(Metal / Hard Rock), or if multiple clips are from the same song. This could result in the same song ending up in both the training set and the testing set. As for technical risks, Training such a large data set requires that we use a GPU. Neither of our laptops have that kind of computing power. Our solution is to use the schools HPC Turing computer, we will access it through either VPN or SSH. If our plan with the spectrographs happens to fall through, we will attempt to use the waveform data of songs/clips. This will allow us to continue to solve the issue we want with a different set of data. 

= Team Responsibilities

Our Responsibilities will be split evenly according to our current plan, this division of tasks may be split further and may change depending on how the project is going. Both members of our current team are student athletes, one of which is in season. Our responsibilities will be changing as circumstance does. 

=== Table of Responsibilities 
#table(
  columns: (auto, 1fr),
  inset: 8pt,
  table.header[*Member*][*Responsibilities*],
  [Joseph Tully], [Data preprocessing, Data Pipe-line, evaluation, report writing],
  [Liam Laidlaw], [Data splitting, ResNet model, evaluation, report writing],
)



= Timeline
#image("gantt.png", width: 100%)  
#pagebreak()
#bibliography("sources.bib")
