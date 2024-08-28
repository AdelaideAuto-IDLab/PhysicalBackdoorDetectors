#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

#import constants as const
import transformations.noise as noise
import transformations.shadow_ellipse as ellipse
#import transformations.shadow_polygon as polygon
import transformations.shadow_single as single


def add_n_random_shadows(image, n_shadow  = 4, blur_scale = 1.0, MIN_SHADOW = 0.2, MAX_SHADOW = 0.6):
	intensity = np.random.uniform(MIN_SHADOW, MAX_SHADOW)
	return add_n_shadows(image, n_shadow, intensity, blur_scale)


def add_n_shadows(image, n_shadow = 4, intensity = 0.5, blur_scale = 1.0):
	for i in range(n_shadow ):

		choice = np.random.uniform(0, 6)
		if choice < 2:
			blur_width = noise.get_blur_given_intensity(intensity, blur_scale)
			image = single.add_single_shadow(image, intensity, blur_width)
		if choice < 4:
			blur_width = noise.get_blur_given_intensity(intensity*0.25, blur_scale)
			image = single.add_single_light(image, intensity*0.25, blur_width)
		if choice < 6:
			blur_width = noise.get_blur_given_intensity(intensity, blur_scale)
			image = ellipse.add_ellipse_shadow(image, intensity, blur_width)

	return image
