# Skin Cancer 
## Overview

Another more interesting than digit classification dataset to use to get biology and medicine students more excited
about machine learning and image processing.

Original Data Source

    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
    Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic 
    images of common pigmented skin lesions. Sci. Data 5, 180161 (2018). doi: 10.1038/sdata.2018.161


Training of neural networks for automated diagnosis of pigmented skin lesions is hampered by the small
size and lack of diversity of available dataset of dermatoscopic images. We tackle this problem by releasing 
the HAM10000 ("Human Against Machine with 10000 training images") dataset. We collected dermatoscopic images 
from different populations, acquired and stored by different modalities. The final dataset consists of 10015 
dermatoscopic images which can serve as a training set for academic machine learning purposes.

## Data 
Cases include a representative collection of all important diagnostic categories in the realm of pigmented 
lesions: 
- Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses
- bkl
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic nevi (nv) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).

More than 50% of lesions are confirmed through histopathology (histo), the ground truth for the rest of the
cases is either follow-up examination (follow_up), expert consensus (consensus), or confirmation by in-vivo
confocal microscopy (confocal). The dataset includes lesions with multiple images, which can be tracked by
the lesion_id-column within the HAM10000_metadata file.

## Code