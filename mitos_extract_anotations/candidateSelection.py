import cv2

from common import utils
from common.Params import Params as P
from mitos_extract_anotations import ImCutter as cut


def join_path(path, add):
    if path[len(path) - 1] != '/':
        return path + '/' + add
    else:
        return path + add


class Candidates_extractor_params:
    def __init__(self, file_list):
        self.save_candidates_dir_path = P().saveCutCandidatesDir
        self.save_mitosis_dir_path = P().saveCutMitosDir
        self.file_list = file_list
        self.candidates_json_save_path = P().candidatesTrainingJsonPath
        self.img_with_keypoints_save_path = P().saveCandidatesWholeImgDir
        self.candidates_save_as_tar = True
        self.mitosis_save_as_tar = False
        self.write_img_to_disk = True
        self.save_img_with_keypoints = True

class Candidates_extractor:
    def __init__(self, params):
        assert isinstance(params, Candidates_extractor_params)
        self.params = params
        self.candidate_cutter = cut.ImCutter(params.save_candidates_dir_path, save_as_tar=params.candidates_save_as_tar)
        if params.save_mitosis_dir_path is not None:
            self.mitos_cutter = cut.ImCutter(params.save_mitosis_dir_path,
                                             save_as_tar=params.mitosis_save_as_tar)
        else:
            self.mitos_cutter = None

        self.verificator = utils.MitosVerification()
        self.candidates_count = 0
        self.candidates_dict = {}

    def extract(self):
        progress = 0
        detector = utils.createBlobDetector()

        utils.print_progress_bar(0, len(self.params.file_list))

        for file in self.params.file_list:
            candidate_list = []
            progress += 1
            color_image = cv2.imread(file.absoluteFilePath())
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
            keypoints = detector.detect(gray_image)

            self.candidate_cutter.set_image(color_image, file.baseName())
            if self.mitos_cutter is not None:
                self.mitos_cutter.set_image(color_image, file.baseName())
                self.verificator.set_base_name(file.baseName())

            candidates, mitos = self._extract_candidates_position(keypoints, file.baseName())
            self.candidates_dict[file.baseName()] = candidates

            if self.params.write_img_to_disk:
                mitos_list = self.verificator.get_mitos_points(file.baseName())
                self._write_img_disk(candidates, mitos_list)

            if self.params.save_img_with_keypoints:
                self.__save_img_with_key_points(color_image, keypoints)

            if self.mitos_cutter is not None:
                self.verificator.print_verification_result()
            utils.print_progress_bar(progress, len(self.params.file_list))

        self.finish()


    def _extract_candidates_position(self, keypoints, base_name):
        candidates_list = []
        mitosis_list = []
        for k in keypoints:
            point = (int(k.pt[0]), int(k.pt[1]))

            if self.mitos_cutter is not None and self.verificator.is_mitos(point):
                mitosis_list.append(point)
                #continue
            else:
                candidates_list.append({"row": point[0], "col": point[1]})
                self.candidates_count += 1

        return candidates_list, mitosis_list

    def _write_img_disk(self, candidates_list, mitosis_list):
        for point in candidates_list:
            self.candidate_cutter.cut_and_save(point["col"], point["row"])

        if self.mitos_cutter is not None:
            for point in mitosis_list:
                self.mitos_cutter.cut_and_save(point["col"], point["row"])

    def __save_img_with_key_points(self, image, keypoints):
        im_with_keypoints = image.copy()
        base_name = self.candidate_cutter._base_name + '.jpg'
        save_path = join_path(self.params.img_with_keypoints_save_path, base_name)
        for k in keypoints:
            point = (int(k.pt[0]), int(k.pt[1]))
            size = int(k.size)
            cv2.circle(im_with_keypoints, point, size, (0,0,255), 2)

        not_found_mitos = self.verificator.not_found_points
        for p in not_found_mitos:
            point = (p['col'], p['row'])
            size = 30
            cv2.circle(im_with_keypoints, point, size, (0,255,0), 2)

        cv2.imwrite(save_path, im_with_keypoints)

    def finish(self):
        self.candidate_cutter.close_tar()
        self.mitos_cutter.close_tar()
        print('Total de candidatos: {}'.format(self.candidates_count))


if __name__ == "__main__":
    filter = ['*.bmp', '*.png', '*.jpg']
    file_list = utils.listFiles(P().normHeStainDir, filter)
    train_list = file_list[0:30]
    test_list = file_list[-5:]
    p = Candidates_extractor_params(train_list)
    c = Candidates_extractor(p)
    c.extract()

    # extract testing dataset
    param = Candidates_extractor_params(test_list)
    param.save_candidates_dir_path = P().saveTestCandidates
    param.save_mitosis_dir_path = P().saveTestMitos
    c = Candidates_extractor(param)
    c.extract()
