import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from depth_encoding import PROTOCOL, brea_geometry_image, continuous_brea_depth, load_profile


def profile():
    return {'protocol':PROTOCOL,'reference_split':'train','native_size':128,
            'smoothing_sigma':0, 'output_smoothing_sigma':0,
            'quantiles':np.linspace(0,1,257).tolist()}


def test_smooth_ramp_is_near_bright_and_not_128_rank_bins():
    depth=np.linspace(2,40,128**2,dtype=np.float32).reshape(128,128)
    result=continuous_brea_depth(depth,profile=profile())
    assert np.isfinite(result).all()
    assert len(np.unique(result)) > 1000
    assert np.all(np.diff(result.ravel()) < 0)
    assert result[0,0] > .99 and result[-1,-1] < .01


def test_masked_pixels_cannot_change_valid_depth_condition():
    depth=np.linspace(2,40,128**2,dtype=np.float32).reshape(128,128)
    valid=np.ones_like(depth,bool); valid[:10]=False
    first=continuous_brea_depth(depth,valid,profile())
    depth[:10]=np.nan
    np.testing.assert_array_equal(first,continuous_brea_depth(depth,valid,profile()))
    depth[:10]=1e8
    np.testing.assert_array_equal(first,continuous_brea_depth(depth,valid,profile()))


def test_constant_geometry_has_no_invented_spatial_features():
    result=continuous_brea_depth(np.full((128,128),7,np.float32),profile=profile())
    np.testing.assert_array_equal(result,np.full((128,128),.5,np.float32))


def test_batch_or_output_resolution_does_not_affect_native_geometry():
    depth=np.linspace(2,40,128**2,dtype=np.float32).reshape(128,128)
    first=brea_geometry_image(depth,profile=profile())
    second=brea_geometry_image(depth,profile=profile())
    np.testing.assert_array_equal(first,second)
    assert first.shape==(256,256) and first.dtype==np.uint8
    assert brea_geometry_image(depth,size=512,profile=profile()).shape==(512,512)


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError,match='no valid'):
        continuous_brea_depth(np.zeros((128,128)),profile=profile())
    with pytest.raises(ValueError,match='match CT depth'):
        continuous_brea_depth(np.ones((128,128)),np.ones((2,2)),profile())


def test_reference_profile_must_be_training_only(tmp_path):
    p=profile();p['reference_split']='test'
    path=tmp_path/'profile.json';path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='training references only'):
        load_profile(path)
    p['reference_split']='train';p['quantiles'][12]=10
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='monotone'):
        load_profile(path)
