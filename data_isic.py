import numpy as np
import pandas as pd


def rotate_z(theta, x):
    theta = np.expand_dims(theta, 1)
    outz = np.expand_dims(x[:, :, 2], 2)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    xx = np.expand_dims(x[:, :, 0], 2)
    yy = np.expand_dims(x[:, :, 1], 2)
    outx = cos_t * xx - sin_t * yy
    outy = sin_t * xx + cos_t * yy
    return np.concatenate([outx, outy, outz], axis=2)


def augment(x):
    bs = x.shape[0]
    # rotation
    min_rot, max_rot = -0.1, 0.1
    thetas = np.random.uniform(min_rot, max_rot, [bs, 1]) * np.pi
    rotated = rotate_z(thetas, x)
    # scaling
    min_scale, max_scale = 0.8, 1.25
    scale = np.random.rand(bs, 1, 3) * (max_scale - min_scale) + min_scale
    return rotated * scale


def standardize(x):
    clipper = np.mean(np.abs(x), (1, 2), keepdims=True)
    z = np.clip(x, -100 * clipper, 100 * clipper)
    mean = np.mean(z, (1, 2), keepdims=True)
    std = np.std(z, (1, 2), keepdims=True)
    return (z - mean) / std


class DataLoaderISIC:
    def __init__(self, df_name, gt_name, batch_size, input_dim=20):
        self.batch_size = batch_size
        self.input_dim = input_dim

        self.gt = pd.read_csv(gt_name)
        self.train_features = pd.read_csv(df_name)

        matching_patient_ids = self.gt[self.gt["image_name"].isin(self.train_features["image_name"])]["patient_id"]
        self.patients = matching_patient_ids.unique()
        self.n_features = sum(['features' in col for col in self.train_features.columns])

    def train_data(self):
        start = 0
        end = self.batch_size
        while end < len(self.patients):
            current_patients = self.patients[start:end]
            current_images = [self.gt[self.gt["patient_id"] == pid]["image_name"].values for pid in current_patients]
            current_features = []
            current_labels = []
            for pnum, images in enumerate(current_images):
                current_patient_features = []
                current_patient_labels = []
                for i in range(self.input_dim):
                    if i < len(images):  # If a patient has more images than needed
                        current_patient_features.append(
                            self.train_features[self.train_features["image_name"] == images[i]].filter(like="feature").values[0])
                        current_patient_labels.append(self.gt[self.gt["image_name"] == images[i]]["target"].values[0])
                        # current_patient_labels.append(np.eye(2)[self.gt[self.gt["image_name"] == images[i]]["target"].values[0]])
                    else:
                        current_patient_features.append(np.zeros(self.n_features))
                        current_patient_labels.append(0)  # TODO not sre whether I should do something else
                        # current_patient_labels.append([1, 0])
                current_features.append(current_patient_features)
                current_labels.append(current_patient_labels)
            current_features = np.asarray(current_features)
            current_labels = np.asarray(current_labels)
            yield current_features, current_labels
            end += self.batch_size
            start += self.batch_size


class ModelFetcherISIC(object):
    def __init__(self, train_name, val_name, gt_name, batch_size, do_standardize=True, do_augmentation=False):
        self.train_name = train_name
        self.batch_size = batch_size

        # Read image names and patient IDs from the train CSV
        train_data = pd.read_csv(train_name)
        self._train_data = train_data.filter(like="feature", axis=1).values
        self._train_data = train_data["target"].values
        self._train_image_names = train_data["image_name"].values

        # Read the ground truth CSV to map image names to patient IDs
        gt_data = pd.read_csv(gt_name)
        self.image_to_patient = dict(zip(gt_data["image_name"], gt_data["patient_id"]))

        # Group image names by patient ID
        self.patient_images = {}
        for image_name, patient_id in self.image_to_patient.items():
            if patient_id not in self.patient_images:
                self.patient_images[patient_id] = []
            self.patient_images[patient_id].append(image_name)

        self.num_classes = np.max(train_data["target"]) + 1

        self.num_train_batches = len(self._train_data) // self.batch_size

        self.prep1 = standardize if do_standardize else lambda x: x
        self.prep2 = (lambda x: augment(self.prep1(x))) if do_augmentation else self.prep1

        self.current_patient = 0

        assert len(self._train_data) > self.batch_size, 'Batch size larger than the number of training examples'

    def train_data(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self._train_data)
        np.random.set_state(rng_state)
        self.current_patient = 0  # Initialize the current patient
        return self.next_train_batch()

    def next_train_batch(self):
        start = 0
        end = self.batch_size
        N = len(self._train_data)
        while end < N:
            # Get the image names for the current patient
            patient_id = list(self.patient_images.keys())[self.current_patient]
            image_names = self.patient_images[patient_id]

            # Get the data and labels for the current batch
            batch_data = [self.prep2(self._train_data[start:end]) for start, end in
                          zip(range(start, end), range(start + self.batch_size, end + self.batch_size))]
            batch_labels = self._train_label[start:end]

            yield batch_data, batch_labels

            # Move to the next patient
            self.current_patient = (self.current_patient + 1) % len(self.patient_images)

# class ModelFetcherISIC(object):
#     def __init__(self, train_name, val_name, batch_size, do_standardize=True, do_augmentation=False):
#
#         self.train_name = train_name
#         self.val_name = val_name
#         self.batch_size = batch_size
#
#         self._train_data = np.asarray(pd.read_csv(train_name).filter(like="feature", axis=1).values)
#         self._train_label = pd.read_csv(train_name)["target"]
#         self._test_data = np.asarray(pd.read_csv(val_name).filter(like="feature", axis=1).values)
#         self._test_label = pd.read_csv(val_name)["target"]
#
#         self.num_classes = np.max(self._train_label) + 1
#
#         self.num_train_batches = len(self._train_data) // self.batch_size
#         self.num_test_batches = len(self._test_data) // self.batch_size
#
#         self.prep1 = standardize if do_standardize else lambda x: x
#         self.prep2 = (lambda x: augment(self.prep1(x))) if do_augmentation else self.prep1
#
#         assert len(self._train_data) > self.batch_size, \
#             'Batch size larger than number of training examples'
#
#     def train_data(self):
#         rng_state = np.random.get_state()
#         np.random.shuffle(self._train_data)
#         np.random.set_state(rng_state)
#         np.random.shuffle(self._train_label)
#         return self.next_train_batch()
#
#     def next_train_batch(self):
#         start = 0
#         end = self.batch_size
#         N = len(self._train_data)
#         while end < N:
#             yield self.prep2(self._train_data[start:end]), self._train_label[start:end]
#             start = end
#             end += self.batch_size
#
#     def test_data(self):
#         return self.next_test_batch()
#
#     def next_test_batch(self):
#         start = 0
#         end = self.batch_size
#         N = len(self._test_data)
#         while end < N:
#             yield self.prep1(self._test_data[start:end]), self._test_label[start:end]
#             start = end
#             end += self.batch_size
