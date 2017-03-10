# all preprocessing code
import dicom
import os
import numpy as np
import scipy.ndimage

from skimage import measure, morphology, segmentation
from functools import partial

def get_slices(path):
    """
    PARAMETERS
    ----------
    path : folder path to a patient

    RETURNS
    -------
    slices : slices for a patient
    """
    # credit: Guido Zuidhof
    # source: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    # if the dicom tag 'scan3D Position Patient' is not found
    # use the the 'Slice Location' tag instead
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        # TODO: verify the 'scan3D Position Patient' & 'Slice Location'
        # are the same for all of the slices for a CT scan
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(slices):
    """
    https://en.wikipedia.org/wiki/Hounsfield_scale

    PARAMETERS
    ----------
    slices : dicom objects of each slice / scan of a patient

    RETURNS
    -------
    image3D : matrix of original data from each slices pixel_array stacked to
        create a matrix in three dimensions
    """
    # credit: Guido Zuidhof
    # source: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook

    # convert to a 3D image
    image3D = np.stack([s.pixel_array for s in slices])

    # convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image3D = image3D.astype(np.int16)

    # set outside-of-scan pixels to 0
    # the intercept is usually -1024, so air is approximately 0
    image3D[image3D == -2000] = 0

    # convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        # http://dicomlookup.com/lookup.asp?sw=Ttable&q=C.7.6.16-10
        # Rescale Type isn't specified as tag on dicom obj, but it
        # is assumed to be HU or UH
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image3D[slice_number] = slope * image3D[slice_number].astype(np.float64)
            image3D[slice_number] = image3D[slice_number].astype(np.int16)

        image3D[slice_number] += np.int16(intercept)

    return np.array(image3D, dtype=np.int16)

def minIP(image3D, slices, minIP_thickness = 6):
    """
    Minimum intensity projection (MinIP) Algorithm as described by...
    https://radiopaedia.org/articles/minimum-intensity-projection-minip

    # mayoclinic states that lung nodules are usually (5 millimeters) to 1.2 inches (30 millimeters)
    # http://www.mayoclinic.org/diseases-conditions/lung-cancer/expert-answers/lung-nodules/faq-20058445
    minIP_thickness = 6
    """
    # TODO: verify this is correct
    slice_diffs = [float(sl.SliceThickness) for sl in slices]

    # create copy s.t. we can compare the original matrix to the new one
    image3D = image3D.copy()

    # TODO: do we care about the last n_slices: throw out?
    # iterate from the top of the lung taking the minimum over the current
    # slices and a number of slices det. by minIP_thickness
    for slice_index in range(len(image3D)):
        # determine the number of slices to search through...
        # * sometimes we'll be missing a slice and we'll use one or two less slices
        # * other times we'll have thicker slices (generally) so we'll also use fewer slices
        slice_end_index = slice_index
        thickness = 0
        while thickness + slice_diffs[slice_end_index] <= minIP_thickness:
            thickness += slice_diffs[slice_end_index]
            slice_end_index += 1
            # since we're no longer useing n_slices to determine
            # when to stop, must break before going out of range
            if slice_end_index + 1 >= len(image3D):
                break
        image3D[slice_index] = np.amin(image3D[slice_index:slice_end_index], axis = 0)

    return image3D

def resample(image3D, slices, new_spacing=[1,1,1]):
    """
    A scan may have a pixel spacing of [2.5, 0.5, 0.5], which means that the distance
    between slices is 2.5 millimeters. For a different scan this may be [1.5, 0.725, 0.725],
    this can be problematic for automatic analysis (e.g. using ConvNets)!

    PARAMETERS
    ----------
    image3D : dicom objects of each slice / scan of a patient

    slices : dicom objects of each slice / scan of a patient

    new_spacing : conversion from current image w/ HU and heterogenious spacing to
        homogenious spacing w/ [x, y, z] voxels

    RETURNS
    -------
    resampled_image3D : image w/ homogenious spacing in [x, y, z] voxels
    """
    # credit: Guido Zuidhof
    # source: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook
    spacing = np.array([slices[0].SliceThickness] + slices[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image3D.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image3D.shape
    new_spacing = spacing / real_resize_factor

    resampled_image3D = scipy.ndimage.interpolation.zoom(image3D, real_resize_factor, mode='nearest')

    # remove return of new_spacing
    return resampled_image3D

MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image3D):
    """
    The values of the pixels currently range from -1024 to 2000. Anything above
    400 is not interesting to us since these are bones. A commonly used set of
    thresholds is to normalize between -1000 and 400.
    """
    # credit: Guido Zuidhof
    # source: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook
    image3D = (image3D - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image3D[image3D > 1] = 1.
    image3D[image3D < 0] = 0.
    return image3D

# FROM LUNA16 competition
PIXEL_MEAN = 0.25

def zero_center(image3D):
    # credit: Guido Zuidhof
    # source: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook
    image3D = image3D - PIXEL_MEAN
    return image3D

def generate_markers(image):
    # shape from images
    x_pixels, y_pixels = image.shape

    # Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    # Creation of the external Marker
    external_a = scipy.ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = scipy.ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    # Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((x_pixels, y_pixels), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed

def separate_lungs(image3D):
    # I thought this would not mess up the original image, but it does.
    # TODO: make a better copy if we want to preserve the original image
    new_image3D = np.copy(image3D)
    for i, image in enumerate(image3D):
        # Creation of the markers as shown above:
        marker_internal, marker_external, marker_watershed = generate_markers(image)

        #Creation of the Sobel-Gradient
        sobel_filtered_dx = scipy.ndimage.sobel(image, 1)
        sobel_filtered_dy = scipy.ndimage.sobel(image, 0)
        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
        sobel_gradient *= 255.0 / np.max(sobel_gradient)

        # Watershed algorithm
        watershed = morphology.watershed(sobel_gradient, marker_watershed)

        # Reducing the image created by the Watershed algorithm to its outline
        outline = scipy.ndimage.morphological_gradient(watershed, size=(3,3))
        outline = outline.astype(bool)

        # Performing Black-Tophat Morphology for reinclusion
        # Creation of the disk-kernel and increasing its size a bit
        blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1, 0, 0]]
        blackhat_struct = scipy.ndimage.iterate_structure(blackhat_struct, 8)
        #Perform the Black-Hat
        outline += scipy.ndimage.black_tophat(outline, structure=blackhat_struct)

        #Use the internal marker and the Outline that was just created to generate the lungfilter
        lungfilter = np.bitwise_or(marker_internal, outline)
        #Close holes in the lungfilter
        #fill_holes is not used here, since in some slices the heart would be reincluded by accident
        lungfilter = scipy.ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)

        #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
        # replaced w/ (x_pixels, y_pixels)
        x_pixels, y_pixels = image.shape

        # used to be segmented...
        lung = np.where(lungfilter == 1, image, -2000 * np.ones((x_pixels, y_pixels)))
        new_image3D[i] = lung
    return new_image3D

def get_preprocessed_patient(path):
    return np.load(path)

def preprocess(path):
    # given a path, run the entire preprocessing code for a patient
    slices = get_slices(path)

    # get dimensions
    # median_dim_z, median_dim_y = get_crop_dimensions(slices)

    # unfortunately resample req. original slices...
    resample_partial = partial(resample, slices = slices, new_spacing=[2, 2, 2])
    minIP_partial = partial(minIP, slices = slices)
    save_partial = partial(save, filename=path.replace('stage1', 'stage1-intermediate-large') + '.npy')
    #crop_partial = partial(crop, median_dim_z, median_dim_y)
    # path should exist

    # determine order for everything
    functions = [
        get_pixels_hu
        #, minIP_partial
        , resample_partial
        , normalize
        , zero_center
        , save_partial
        #, crop_partial # don't know if this would work here
    ]

    return reduce(lambda x, y: y(x), functions, slices)

def get_crop_dimensions(images3D):
    # get the mean in the dimensions
    # [images3D.shape(); for image in]
    dimensions = []
    for image3D in images3D:
        cur_z, cur_y, _ = image3D.shape
        dimensions.append((cur_z, cur_y))

    dims_z, dims_y = zip(*dimensions)

    # median is easier out of the box
    median_dim_z = int(np.median(dims_z))
    median_dim_y = int(np.median(dims_y))

    return(median_dim_z, median_dim_y)

def crop(image3D, median_dim_z, median_dim_y):
    image_z, image_y, _ = image3D.shape
    diff_z = image_z - median_dim_z
    diff_y = image_y - median_dim_y

    padding_z = (0, 0)
    padding_y = (0, 0)

    # pad the image
    if diff_z < 0:
        padding_z_top = abs(diff_z) // 2
        padding_z_bottom = abs(diff_z) - padding_z_top
        padding_z = (padding_z_top, padding_z_bottom)

    if diff_y < 0:
        padding_y_top = abs(diff_y) // 2
        padding_y_bottom = abs(diff_y) - padding_y_top
        padding_y = (padding_y_top, padding_y_bottom)

    if diff_z < 0 or diff_y < 0:
        padding = (padding_z, padding_y, padding_y)

        # :( padding can't take negative values
        # update the image w/ padded image
        # need to create copy, b.c. it's inplace op.
        # pasdding with -0.25 since it'll be the minimum pixel value
        image3D = np.pad(np.copy(image3D), padding, 'constant', constant_values=-PIXEL_MEAN)
        image_z, image_y, _ = image3D.shape


    # crop the image
    crop_z_top = 0
    crop_z_bottom = image_z
    crop_y_top = 0
    crop_y_bottom = image_y

    if diff_z > 0:
        crop_z_top = diff_z // 2
        crop_z_bottom = -(diff_z - crop_z_top)

    if diff_y > 0:
        crop_y_top = diff_y // 2
        crop_y_bottom = -(diff_y - crop_y_top)

    if diff_z > 0 or diff_y > 0:
        cropped_image = np.copy(image3D)

        cropped_image = cropped_image[
            crop_z_top:crop_z_bottom,
            crop_y_top:crop_y_bottom,
            crop_y_top:crop_y_bottom
        ]

        return cropped_image
    else:
        return image3D

def save(image3D, filename):
    np.save(filename, image3D)
    return image3D

