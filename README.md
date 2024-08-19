# Protocol-for-deriving-longitudinal-Clinical-Outcomes-from-electronic-health-records

This repository is a summary of all the resources and tools that includes in this protocol: Protocol for deriving longitudinal Clinical Outcomes from electronic health records.

## Contents

## Description and flowchart

In this protocol, we are going to present a method to capture longitudinal clinical outcomes from structured and unstructured electronic health record based on the the high-throughput **M**ultimodal **A**utomated **P**henotyping **(MAP)** algorithm and **LA**bel efficien**T** inciden**T** ph**E**notyping **(LATTE)** algorithm . The result, which is the qualified clinical outcomes can be used to harness electronic health data to real-world data. The protocol is generally made up by 4 steps:

1)  Creating EHR Data mart;
2)  Selecting codified and narrative features with ONCE, compiling and performing the MAP algorithm;
3)  Preparing data for LATTE algorithm;
4)  Performing the LATTE to get longitudinal outcome.

![**Figure 1:** The pipeline of the longitudinal clinical outcomes deriving from EHR](Flowchart.png)

## Method

### Step 1: Creating EHR Data mart

The EHR Data mart is the basic component of the protocol, it consists both the codified data and narrative data. The raw EHR data includes a great amount of information, and many of them do not need to be involve in our study taken both confidentiality and storage availability under consideration. We need a variable dictionary for our specific study on the target disease. The shiny app ONCE can help us derive the relative codified features(PheCode, CCS, LOINC, RxNorm) and narrative features(relative CUIs and terms) .

| Use                           | Method | Links                                         | References                                                                                                                       |
|-------------------------------|--------|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Compiling Variable Dictionary | ONCE   | [ONCE](https://shiny.parse-health.org/ONCE/#) | [Knowledge-Driven Online Multimodal Automated Phenotyping System](https://www.medrxiv.org/content/10.1101/2023.09.29.23296239v1) |
