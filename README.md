# JINSv2__Thesis

JINS (JINS Is Not A Simulator) generate an artifical scene of a satellite in LEO orbit. The renedered images are used to generate a COCO dataset for the training of 2 Neural Network for instance segmentation :  Mask R-CNN and PointRend (both deetctron2).

>> 1 : 'Render_final.py' is the main script and generate the scene (satellite + Earth) and the binary mask of each component shown in the scene.
        It generates a mask for each label, in my case only 4 labels are used : 'antenna' , 'body', 'solarPanel', 'Earth'.
        For the generation of the scene satellite's blender model (contact me if needed).
        
>> 2 : 'satellitetoCOCO.py' is used to generate a COCO dataset. Once runned it generates a folder called 'annotations' inside which are 3 .json files.
        Depending on the operative mode used (default : training) it filled the .json files.
        
>> 3 : 'BBox.py' is used to verify if the previous step has generated a correct bounding box and the correcty polygon (mask for segemntation).

>> 4 : The .ipynb files for the neural network will be loaded later.
