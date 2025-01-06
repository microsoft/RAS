class ras_manager:
    def __init__(self):
        self.patch_size = 2
        self.scheduler_start_step = 4
        self.scheduler_end_step = 30
        self.metric = "std"
        self.error_reset_steps = [12, 22]
        self.replace_with_flash_attn = False
        self.sample_ratio = 0.5
        self.starvation_scale = 0.1
        self.vae_size = 8
        self.high_ratio = 1
        self.num_steps = 30
        self.current_step = 0
        self.enable_index_fusion = False
        self.is_RAS_step = False
        self.is_next_RAS_step = False

        self.cached_index = None
        self.other_index = None
        self.cached_patchified_index = None
        self.other_patchified_index = None
        self.image_rotary_emb_skip = None
        self.cached_scaled_noise = None
        self.skip_token_num_list = []


    def set_parameters(self, args):
        self.patch_size = args.patch_size
        # self.scheduler_pattern = args.scheduler_pattern
        self.scheduler_start_step = args.scheduler_start_step
        self.scheduler_end_step = args.scheduler_end_step
        self.metric = args.metric
        self.error_reset_steps = [int(i.strip()) for i in args.error_reset_steps.split(",")]
        self.replace_with_flash_attn = args.replace_with_flash_attn
        self.sample_ratio = args.sample_ratio
        self.num_steps = args.num_inference_steps
        self.skip_num_step = args.skip_num_step
        self.skip_num_step_length = args.skip_num_step_length
        self.height = args.height
        self.weight = args.width
        self.high_ratio = args.high_ratio
        self.enable_index_fusion = args.enable_index_fusion
        self.generate_skip_token_list()


    def generate_skip_token_list(self):
        avg_skip_token_num = int((1 - self.sample_ratio) * ((self.height // self.patch_size) // self.vae_size) * ((self.weight // self.patch_size) // self.vae_size))
        if self.skip_num_step_length == 0: # static dropping
            self.skip_token_num_list = [avg_skip_token_num for i in range(self.num_steps)]
            for i in self.error_reset_steps:
                self.skip_token_num_list[i] = 0
            for i in range(self.scheduler_start_step):
                self.skip_token_num_list[i] = 0
            print("Skip token number list: ", len(self.skip_token_num_list), self.skip_token_num_list)
            return
        for i in range(0, self.num_steps // self.skip_num_step_length + 1):
            for j in range(self.skip_num_step_length):
                if i * self.skip_num_step_length + j >= self.num_steps:
                    break
                temp_skip_num = avg_skip_token_num + self.skip_num_step * (i - (((self.num_steps + self.scheduler_start_step) // self.skip_num_step_length) // 2))
                temp_skip_num = (temp_skip_num // 64) * 64
                self.skip_token_num_list.append(temp_skip_num)
        for i in range(self.scheduler_start_step):
            self.skip_token_num_list[i] = 0
        for i in self.error_reset_steps:
            self.skip_token_num_list[i] = 0
        for i in range(len(self.skip_token_num_list)):
            assert self.skip_token_num_list[i] >= 0, "Skip token number should be positive"
            assert self.skip_token_num_list[i] <= ((self.height // self.patch_size) // self.vae_size) * ((self.weight // self.patch_size) // self.vae_size)
        print("Skip token number list: ", len(self.skip_token_num_list), self.skip_token_num_list)

    def reset_cache(self):
        self.cached_index = None
        self.other_index = None
        self.cached_patchified_index = None
        self.other_patchified_index = None
        self.image_rotary_emb_skip = None
        self.cached_scaled_noise = None
        self.current_step = 0
        if self.current_step >= self.scheduler_start_step and self.current_step <= self.scheduler_end_step and self.current_step not in self.error_reset_steps:
            self.is_RAS_step = True
        else:
            self.is_RAS_step = False
        if self.current_step + 1 >= self.scheduler_start_step and self.current_step + 1 <= self.scheduler_end_step and self.current_step + 1 not in self.error_reset_steps:
            self.is_next_RAS_step = True
        else:
            self.is_next_RAS_step = False

    def increase_step(self):
        self.current_step += 1
        if self.current_step >= self.scheduler_start_step and self.current_step <= self.scheduler_end_step and self.current_step not in self.error_reset_steps:
            self.is_RAS_step = True
        else:
            self.is_RAS_step = False
        if self.current_step + 1 >= self.scheduler_start_step and self.current_step + 1 < self.scheduler_end_step and self.current_step + 1 not in self.error_reset_steps:
            self.is_next_RAS_step = True
        else:
            self.is_next_RAS_step = False







MANAGER = ras_manager()
