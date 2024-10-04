# Converting volumes to nii.gz
for volumeNode in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
	volumeNode.AddDefaultStorageNode()
	slicer.util.saveNode(volumeNode, "/Volumes/Extreme SSD/ANTs-registration/images/NIFTI Images/" + volumeNode.GetName() + ".nii.gz")


# Converting segmentations to nii.gz
volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
segmentations = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")

i = 0
for segmentationNode in segmentations:
	print(i)
	volume_name = ' '.join(segmentationNode.GetName().split(' ')[1:])
	print(volume_name)
	try:
		referenceVolumeNode = getNode(volume_name)
	except:
		continue
	labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
	slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segmentationNode, labelmapVolumeNode, referenceVolumeNode)
	labelmapVolumeNode.AddDefaultStorageNode()
	slicer.util.saveNode(labelmapVolumeNode, "/Volumes/Extreme SSD/ANTs-registration/predictions/NIFTI Predictions/" + "Segmentation " + volume_name + ".nii.gz")
	i += 1