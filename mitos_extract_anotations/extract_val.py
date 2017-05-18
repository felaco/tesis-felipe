from common import utils
from mitos_extract_anotations import candidateSelection as cs

if __name__ == '__main__':
    filter = ['*.bmp', '*.png', '*.jpg']
    file_list = utils.listFiles('C:/Users/felipe/mitos dataset/eval/heStain', filter)
    params = cs.Candidates_extractor_params(file_list)
    params.candidates_json_save_path = 'test_cand.json'
    params.save_candidates_dir_path = 'C:/Users/felipe/mitos dataset/eval/no-mitosis/'
    params.save_mitosis_dir_path = 'C:/Users/felipe/mitos dataset/eval/mitosis/'
    params.bsave_img_keypoints = True

    cutter = cs.Candidates_extractor(params)
    cutter.extract()