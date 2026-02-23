from multiprocessing import Process

import numpy as np
import torch
import torchvision.transforms as transforms

import environment


class Hw1Env(environment.BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        r = np.random.rand()
        if r < 0.5:
            size = np.random.uniform([0.02, 0.02, 0.02], [0.03, 0.03, 0.03])
            environment.create_object(scene, "box", pos=[0.6, 0., 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=[0.8, 0.2, 0.2, 1], friction=[0.02, 0.005, 0.0001],
                                      density=4000, name="obj1")
        else:
            size = np.random.uniform([0.02, 0.02, 0.02], [0.03, 0.03, 0.03])
            environment.create_object(scene, "sphere", pos=[0.6, 0., 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=[0.8, 0.2, 0.2, 1], friction=[0.2, 0.005, 0.0001],
                                      density=4000, name="obj1")
        return scene

    def state(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return obj_pos, pixels

    def step(self, action_id):
        if action_id == 0:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 1:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 2:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 3:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})


def collect(idx, N):
    env = Hw1Env(render_mode="offscreen")

    # AFTER position (label for deliverable 1-2)
    positions = torch.zeros((N, 2), dtype=torch.float32)

    # action id
    actions = torch.zeros((N,), dtype=torch.uint8)

    # BEFORE image (input for deliverable 1-2 and 3)
    imgs_before = torch.zeros((N, 3, 128, 128), dtype=torch.uint8)

    # AFTER image (target for deliverable 3)
    imgs_after  = torch.zeros((N, 3, 128, 128), dtype=torch.uint8)

    for i in range(N):
        env.reset()
        action_id = np.random.randint(4)

        # BEFORE
        _, pixels_before = env.state()

        # STEP
        env.step(action_id)

        # AFTER
        obj_pos_after, pixels_after = env.state()

        # --- robust type conversions (torch / numpy farkı olursa diye) ---
        if isinstance(pixels_before, np.ndarray):
            pixels_before = torch.from_numpy(pixels_before)
        if isinstance(pixels_after, np.ndarray):
            pixels_after = torch.from_numpy(pixels_after)

        pixels_before = pixels_before.to(torch.uint8)
        pixels_after  = pixels_after.to(torch.uint8)

        if isinstance(obj_pos_after, np.ndarray):
            obj_pos_after = torch.from_numpy(obj_pos_after)
        obj_pos_after = obj_pos_after.to(torch.float32)

        # save to buffers
        actions[i] = action_id
        positions[i] = obj_pos_after
        imgs_before[i] = pixels_before
        imgs_after[i]  = pixels_after

        if (i + 1) % 50 == 0:
            print(f"[proc {idx}] {i+1}/{N}", flush=True)

    # Save (isimleri ayır ki karışmasın)
    torch.save(positions,   f"positions_{idx}.pt")
    torch.save(actions,     f"actions_{idx}.pt")
    torch.save(imgs_before, f"imgs_before_{idx}.pt")
    torch.save(imgs_after,  f"imgs_after_{idx}.pt")


if __name__ == "__main__":
    processes = []
    for i in range(4):
        p = Process(target=collect, args=(i, 250))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
