import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

from tts_arabic.models import core
from tts_arabic.models.tts_models import FastPitch2Wave


class TestAPNet2VocoderSupport(unittest.TestCase):
    def test_apnet2_is_advertised_as_available_vocoder(self):
        self.assertIn('apnet2', core.get_available_models()['vocoders'])

    @patch('tts_arabic.models.core.gdown.download')
    def test_apnet2_requires_local_model_file_when_url_is_unset(
            self, mock_download):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                core.get_model_path(Path(tmpdir), 'apnet2')
        mock_download.assert_not_called()

    @patch('tts_arabic.models.core.FastPitch2Wave')
    @patch('tts_arabic.models.core.get_model_path')
    def test_apnet2_model_uses_existing_pipeline_without_denoiser(
            self, mock_get_model_path, mock_fastpitch2wave):
        mock_get_model_path.side_effect = [
            '/tmp/fp_ms.onnx',
            '/tmp/apnet2.onnx',
        ]
        core.get_model(vocoder_id='apnet2', cuda=None)

        mock_fastpitch2wave.assert_called_once_with(
            '/tmp/fp_ms.onnx',
            '/tmp/apnet2.onnx',
            None,
            vocoder_id='apnet2',
            cuda=None,
        )

    @patch('tts_arabic.models.tts_models.VocosVocoder')
    @patch('tts_arabic.models.tts_models.FastPitch2Mel')
    def test_apnet2_uses_vocos_vocoder_pipeline(
            self, mock_fastpitch2mel, mock_vocos):
        FastPitch2Wave(
            sd_path_ttmel='/tmp/fp_ms.onnx',
            sd_path_mel2wave='/tmp/apnet2.onnx',
            sd_path_denoiser='/tmp/denoiser.onnx',
            vocoder_id='apnet2',
            cuda=None,
        )
        mock_fastpitch2mel.assert_called_once_with('/tmp/fp_ms.onnx', cuda=None)
        mock_vocos.assert_called_once_with('/tmp/apnet2.onnx', cuda=None)


if __name__ == '__main__':
    unittest.main()
