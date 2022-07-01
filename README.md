# CLA-HDM
&ensp; &ensp; This work is proposed in [Deep Learning Radiomics of Dual-modality Ultrasound Images for Hierarchical Diagnosis of Unexplained Cervical Lymphadenopathy](), which has been potentially accepted by *BMC Medicine*. It is a hierarchical diagnosis model for unexplained cervical lymphadenopathy or any other diseases based on ultrasound and color Doppler flow images.

&ensp; &ensp; This work is a cooperation achievement by researchers from Lanzhou University Second Hospital and Institute of Automation CAS.

![image](https://user-images.githubusercontent.com/57392333/176902551-2c00d85d-6673-4233-95c5-a6edf35fc584.png)

## Model structure
&ensp; &ensp; The structure of CLA-HDM is shown in figure b. It consists of three sub-models (figure a) and each one takes dual-modality images as input to extract different features for different sub-type levels (benign or malignant, reactive hyperplasia or tuberculous lymphadenitis in benign candidates and of lymphoma or metastatic carcinoma in malignant candidates). RoIs can be segmented by clinicians or lesion segmentation networks (e.g. [CEUSegNet](https://github.com/RichardSunnyMeng/CEUSegNet)).

## Training
&ensp; &ensp; We trained three sub-models seperately and then fine-tuned CLA-HDM on our training dataset.

## Inference
&ensp; &ensp; CLA-HDM determined the specific pathological type based on the diagnosis conclusion of benign and malignant. Specifically, let the probability of malignant by sub-model 1 be $p_1$, the probability of tuberculous and reactive by sub-model 2 be $p_2$(Tuberculous)and $p_2$(Reactive), the probability of lymphoma and metastatic by sub-model 3 be $p_3$(Lymphoma) and $p_3$(Metastatic), respectively. Then the probability for each pathological type by CLA-HDM on each pathological type is:
![1656683208133](https://user-images.githubusercontent.com/57392333/176906930-7aca0503-97df-4a50-9bcf-1363f55fad17.png)

## Experiments
&ensp; &ensp; Multi-cohort testing demonstrated our model integrating dual-modality ultrasound images achieved accurate diagnosis of unexplained CLA. With its assistance, the gap between radiologists with different levels of experience was narrowed, which is potentially of great significance for benefiting CLA patients in underdeveloped countries and regions worldwide.
![image](https://user-images.githubusercontent.com/57392333/176905003-f3aff3b4-6880-4d3b-954c-04f1f1935506.png)

&ensp; &ensp; More details refer to our paper.
