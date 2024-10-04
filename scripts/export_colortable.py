import slicerio
import nrrd
import csv

input_filename = "/Users/andyding/Downloads/Segmentation RT 153.seg.nrrd"
output_filename = "/Users/andyding/Downloads/Segmentation RT 153 Color Table"

voxels, header = nrrd.read(input_filename)
segmentation_info = slicerio.read_segmentation(input_filename)

number_of_segments = len(segmentation_info["segments"])
print(f"Number of segments: {number_of_segments}")

data = [['Segment', 'Color']]

segments = segmentation_info['segments']
for segment in segments:
    data.append([segment['name'], segment['color']])

with open('tbone_colortable.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)