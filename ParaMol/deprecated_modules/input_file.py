import copy

class Settings:
    _SYSTEM_INPUT_PARAMETERS = {"name" : None,
                                "top_file" : None,
                                "top_file": None,
                                "crd_file": None,
                                "n_cpus" : 1,
                                "coords_file" : None,
                                "energies_file": None,
                                "forces_file": None,
                                "mm_engine" : "OpenMM",
                                "qm_engine" : "AMBER",
                                "platform_name" : "Reference",
                                "extra_torsions" : False,
                                "dt": 0.001,
                                "temperature": 300.0,
                                "opt_bonds":  False,
                                "opt_angles": False,
                                "opt_torsions" : False,
                                "opt_lj" : False,
                                "opt_charges" : False,
                                "opt_sc" : False,}

    _OPTIMIZER_SETTINGS = {"method": "SciPy",
                           "scipy_method": "SLSQP",
                           "max_iter":  200000,
                           "f_threshold": 1e-6,
                           "seed": 111111,}

    _OBJECTIVE_FUNCTION_SETTINGS = {"energies": True,
                                    "forces": True,
                                    "regularization": False}

    def __init__(self):
        self.objective_function_settings = None
        self.optimizer_settings = None
        self.systems_settings = []

    def read_system(self, system_file="paramol.systems"):
        """
        Read paramol.systems file.


        :param system_file:
        :return:
        """

        with open(system_file, 'r') as f:
            for line in f:
                if len(line) != 1:
                    if "%start_system" in line.upper():
                        # create new system
                        settings = copy.deepcopy(self._SYSTEM_INPUT_PARAMETERS)
                        self.systems_settings.append(settings)
                    elif "%end_system" in line.upper():
                        continue

        return self.systems_settings

    def read_task(self, system_file="paramol.task"):
        """

        :param system_file:
        :return:
        """

        return self.optimizer_settings, self.objective_function_settings

    def print_user_input(self):
        vars_dict = vars(self)
        print('''
    !---------------------------------------------------------------------------------!
    !                              User Input Parameters                              !
    !---------------------------------------------------------------------------------!
    ''')
        for var in vars_dict:
            print('\t - {} : {}'.format(var,vars_dict[var]))

        print('''
    !---------------------------------------------------------------------------------!
    !                          End Of User Input Parameters                           !
    !---------------------------------------------------------------------------------!
    ''')

        return vars_dict

    @staticmethod
    def print_header():
        import datetime
        print("""
    !---------------------------------------------------------------------------------!
    !                                                                                 !
    !                        _____                __  __       _                      !
    !                       |  __ \              |  \/  |     | |                     !
    !                       | |__) |_ _ _ __ __ _| \  / | ___ | |                     !
    !                       |  ___/ _` | '__/ _` | |\/| |/ _ \| |                     !
    !                       | |  | (_| | | | (_| | |  | | (_) | |                     !
    !                       |_|   \__,_|_|  \__,_|_|  |_|\___/|_|                     !
    !                                    version 0.1                                  !
    !                                                                                 !
    !                          Code by Joao Morado - 2019-201*                        !
    !                                                                                 !
    !---------------------------------------------------------------------------------!""")
        print('\n' * 1)
        print(str.center(str(datetime.datetime.now()), 90))
        return

    def read_input_file(self, input_file):
        with open(input_file, 'r') as f:
            for line in f:
                line = line.replace('\n', '').split("=")
                if len(line) != 1:
                    parameter = line[0].strip()
                    
                    if parameter[0] == "#":
                        # Comment
                        continue
                    else:
                        value = line[1].strip()
                        if parameter.lower() in self.INPUT_PARAMETERS:
                            if value.lower() == 'none':
                                setattr(self, parameter, None)
                            elif value.lower() == 'true':
                                setattr(self, parameter, True)
                            elif value.lower() == 'false':
                                setattr(self, parameter, False)
                            elif value[0] == "'" or value[0] == '"':
                                value = value.replace('"', '')
                                value = value.replace("'", '')
                                setattr(self, parameter, str(value))
                            elif "." in value or "e" in value:
                                setattr(self, parameter, float(value))
                            else:
                                setattr(self, parameter, int(value))
                        else:
                            raise ValueError('Parameter {} does not exist. Check the input file.'.format(parameter))


if __name__ == '__main__':
    pass