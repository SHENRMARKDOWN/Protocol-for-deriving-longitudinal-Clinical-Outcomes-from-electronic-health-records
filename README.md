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

The EHR Data mart is the basic component of the protocol, it consists both the codified data and narrative data.

##### Compiling Variable Dictionary

The raw EHR data includes a great amount of information, and many of them do not need to be involve in our study taken both confidentiality and storage availability under consideration. We need a variable dictionary for our specific study on the target disease. The shiny app ONCE can help us derive the relative codified features(PheCode, CCS, LOINC, RxNorm) and narrative features(relative CUIs and terms) .

| Use                           | Method | Links                                         | References                                                                                                                       |
|-------------------------------|--------|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Compiling Variable Dictionary | ONCE   | [ONCE](https://shiny.parse-health.org/ONCE/#) | [Knowledge-Driven Online Multimodal Automated Phenotyping System](https://www.medrxiv.org/content/10.1101/2023.09.29.23296239v1) |

##### Aggregating codified data

Aggregating codified data Raw EHR concepts are aggregated to higher level concepts.

| Use                      | Method | Links                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | References                                                                                                                                                                   |
|--------------------------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Aggregating EHR concepts | PheWAS | [PheWAS](https://phewascatalog.org/) , [Phecode Map 1.2 with ICD-9 Codes](https://phewascatalog.org/phecodes) , [Phecode Map 1.2 with ICD-10cm Codes (beta)](https://phewascatalog.org/phecodes_icd10cm)                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | [PheWAS: demonstrating the feasibility of a phenome-wide scan to discover gene–disease associations](https://academic.oup.com/bioinformatics/article/26/9/1205/201211)       |
| Aggregating EHR concepts | CCS    | [ICD-9-CM Diagnosis and Procedure Codes: Abbreviated and Full Code Titles](https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-9-cm-diagnosis-procedure-codes-abbreviated-and-full-code-titles) , [2024 ICD-10-PCS](https://www.cms.gov/medicare/coding-billing/icd-10-codes/2024-icd-10-pcs) , [List of CPT/HCPCS Codes](https://www.cms.gov/medicare/regulations-guidance/physician-self-referral/list-cpt-hcpcs-codes) , [Clinical Classifications Software (CCS) for ICD-9-CM](https://hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp) , [Clinical Classifications Software (CCS) for ICD-10-PCS (beta version)](https://hcup-us.ahrq.gov/toolssoftware/ccs10/ccs10.jsp) | [Hospital Inpatient Statistics, 1995](https://hcup-us.ahrq.gov/reports/natstats/his95/clinclas.htm)                                                                          |
| Aggregating EHR concepts | LOINC  | [Download LOINC](https://loinc.org/downloads/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | [LOINC, a universal standard for identifying laboratory observations: a 5-year update](https://pubmed.ncbi.nlm.nih.gov/12651816/)                                            |
| Aggregating EHR concepts | RxNorm | [RxNorm Files](https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | [RxNorm: Prescription for Electronic Drug Information Exchange](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.nlm.nih.gov/research/umls/rxnorm/RxNorm.pdf) |
