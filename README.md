## UNIVPM-INRIA DCASE track 4 Sound Event Detection System code/repo. 

In this repo we have uploaded the code used for DCASE2020 Track 4 Challenge.
 
We achieved as a team the 10th place in the SED category. 

However, regarding class-wise detection performance we placed first for the cat class, thus this repo
can be especially useful for cat lovers :smiley_cat:.

We employed the challenge official baseline architecture based on CNN-RNN and Mean Teacher training.
This allowed us to focus on the **training procedure**, 
on the **feature pre-processing** and on the **prediction post-processing and smoothing**. 
Regarding training procedure, we achieved good validation set results by combining the Mean Teacher with **Domain Adversarial Training**
and online creation of synthetic labeled examples.  
This, further combined with **Hidden Markov Model prediction smoothing** allowed us to achieve 45.2 %event-based macro F1 score 
on the validation set. We also explored feature pre-processing by employing several **parallel 
Per-Channel Energy Normalization front-end layers (PPCEN)**.

####For more information have a look at our [DCASE2020 system description](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Cornell_130.pdf) and to the [official DCASE 2020 Task 4 Challenge results page](http://dcase.community/challenge2020/task-sound-event-detection-and-separation-in-domestic-environments-results) 

---
If you find this code useful please cite:
```BibTex
@article{CornellDCASE2020,
    title={The UNIVPM-INRIA Systems for the  DCASE 2020 TASK 4},
    author={Samuele Cornell, Giovanni Pepe, Emanuele Principi, Manuel Pariente, Michel Olvera, Leonardo Gabrielli, Stefano Squartini},
    year={2020},
    journal={DCASE 2020 Task 4 Challenge System description},
   
}
```





