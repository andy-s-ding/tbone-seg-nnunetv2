import slicerio
import nrrd
import csv
import pandas as pd

input_filename = "/Users/andyding/Downloads/Segmentation RT 153.nii.gz"
output_filename = "/Users/andyding/Downloads/Segmentation RT 153 Test.seg.nrrd"
colortable_filename = "/Users/andyding/Documents/Johns Hopkins/OHNS Research/Automated CT Segmentation/tbone_colortable.csv"

colortable = pd.read_csv(colortable_filename)
segmentation_info = slicerio.read_segmentation(input_filename)

number_of_segments = len(segmentation_info["segments"])
print(f"Number of segments: {number_of_segments}")

data = [['Segment', 'Color']]

segments = segmentation_info['segments']
for i, segment in enumerate(segments):
    segment['name'] = colortable['Segment'][i]
    segment['color'] = [eval(i) for i in colortable['Color'][i][1:-1].split(',')]

slicerio.write_segmentation(output_filename, segmentation_info)