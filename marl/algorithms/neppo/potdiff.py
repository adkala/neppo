import copy

import torch as th


class PotDiffStep:
    def __init__(self, hp, num_actors):
        self.hp = hp
        self.num_actors = num_actors

        self.pm_actors = None
        self.br_actors = None

        self.phi = None

        self.pm_dataset = None
        self.br_datasets = [None] * self.num_actors

        self.i = 0

    def set_networks(self, pm_actors, br_actors, phi):
        self.pm_actors = pm_actors
        self.br_actors = br_actors
        self.phi = phi

        # Skip contiguous for ensemble wrapper - it manages its own parameters
        if not hasattr(self.phi, "phi_ensemble"):
            for param in self.phi.parameters():
                param.data = param.data.contiguous()

        self.pm_dataset = None
        self.br_datasets = [None] * self.num_actors

    def clear_networks(self):
        self.pm_actors = None
        self.br_actors = None
        self.phi = None

    def save_dataset(self, tds, log_prefix="pd"):
        log_dict = {f"{log_prefix}/rollout/agent_idx": self.i}
        iter_flag = False
        if self.i == self.num_actors:
            self.pm_dataset = tds
            rewards = th.stack(
                [tds[j]["next", "reward"] for j in range(self.num_actors)]
            )
            for i in range(self.num_actors):
                self.pm_dataset[i]["rewards_T"] = rewards.T
            self.i = 0
            iter_flag = True
        else:
            self.br_datasets[self.i] = tds[self.i]
            self.br_datasets[self.i]["rewards_T"] = th.stack(
                [tds[j]["next", "reward"] for j in range(self.num_actors)]
            ).T
            self.i += 1

        return (
            log_dict,
            iter_flag,
            tds[0].shape[0],
        )

    def clear_datasets(self):
        self.pm_dataset = None
        self.br_datasets = [None] * self.num_actors

    def fs(self):
        fs = []
        log_dict = {}
        for i in range(self.num_actors):
            td = self.pm_dataset[i]
            br_td = self.br_datasets[i]

            pm_rewards = self.phi(td["state"], td["rewards_T"].T)
            br_rewards = self.phi(br_td["state"], br_td["rewards_T"].T)

            br_Phi = (1 / br_td["next", "done"].sum()) * (
                th.sum(br_rewards * th.pow(self.hp.gamma, br_td["step_count"]))
            )
            Phi = (1 / td["next", "done"].sum()) * (
                th.sum(pm_rewards * th.pow(self.hp.gamma, td["step_count"]))
            )

            br_V = (1 / br_td["next", "done"].sum()) * (
                th.sum(
                    br_td["next", "reward"] * th.pow(self.hp.gamma, br_td["step_count"])
                )
            )
            V = (1 / td["next", "done"].sum()) * (
                th.sum(td["next", "reward"] * th.pow(self.hp.gamma, td["step_count"]))
            )

            log_dict[f"{i}/Phi"] = Phi.item()
            log_dict[f"{i}/br_Phi"] = br_Phi.item()
            log_dict[f"{i}/V"] = V.item()
            log_dict[f"{i}/br_V"] = br_V.item()
            log_dict[f"{i}/delta_Phi (pm - br)"] = (Phi - br_Phi).item()
            log_dict[f"{i}/delta_V (pm - br)"] = (V - br_V).item()

            fs.append((Phi - br_Phi) - (V - br_V))

        return fs, log_dict

    @property
    def actors(self):
        actors = [actor for actor in self.pm_actors]
        if self.i < self.num_actors:
            actors[self.i] = self.br_actors[self.i]
        return actors
