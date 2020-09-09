# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Utils.interface.ParaMolInterface` used for ParaMol to interact with system.
"""

import logging
import subprocess
import os


class ParaMolInterface:
    """
    This class defines method that has, for example, moving between and creating new directories, running subprocesses, etc...

    Parameters
    ----------
    base_dir : str, optional, default=os.getcwd()
        Path to the base working directory.

    Attributes
    ----------
    base_dir : str
        Path to the base working directory.
    dirs : list
        Keeps track of where we are at.
    created_dirs : list
        List of all created directories.
    cwd : list
        Current working directory
    """
    def __init__(self, base_dir=None):
        self.base_dir = os.getcwd() if base_dir is None else base_dir
        self.dirs = [self.base_dir]
        self.cwd = os.getcwd()
        self.created_dirs = []

    def remove_all_created_dirs(self):
        """
        Method that removes all created directories.

        Returns
        -------
        created_dirs: list
            It should be an empty list.
        """

        while len(self.created_dirs) > 0:
            # Current directory
            current_dir = self.created_dirs.pop()
            # Check if directory exists
            self.check_dir_exists(current_dir)
            # Remove dir
            os.rmdir(current_dir)

        return self.created_dirs

    def chdir(self, *args, absolute=False, relative_to_base=False):
        """
        Method that changes directory.

        Parameters
        ----------
        args : strings
            Strings to be joined.
        absolute : bool
            If True it assumes the first argument is absolute and the remaining are are defined relatively to the first.
        relative_to_base : bool, default=False
            If `False` path are defined relatively to the current working directory, if `True` paths are defined relatively to the base working directory.
            Ignored if `absolute` is `True`.

        Returns
        -------
        cwd : str
            Current working directory.
        """
        # Define prefix
        if relative_to_base:
            prefix = self.base_dir
        else:
            prefix = self.cwd

        # Change directory
        if absolute:
            path = os.path.join(*args)
            self.check_dir_exists(path)
            os.chdir(path)
            self.cwd = os.getcwd()
        else:
            path = os.path.join(prefix, *args)
            self.check_dir_exists(path)
            os.chdir(path)
            self.cwd = os.getcwd()

        self.dirs.append(self.cwd)

        return self.cwd

    def chdir_previous(self):
        """
        Method that changes to previous directory.

        Returns
        -------
        cwd : str
            Current working directory.
        """
        if len(self.dirs) == 1:
            if self.dirs == [self.base_dir] and os.getcwd() == self.base_dir and self.base_dir == self.cwd:
                logging.warn("ParaMol cannot change to previous directory as it is already in the base directory.")
            else:
                logging.warning("Something went wrong while trying to change to previous directory.")
                raise NotADirectoryError("Something went wrong while trying to change to previous directory.")
        else:
            os.chdir("..")
            self.dirs.pop()
            self.cwd = os.getcwd()
            assert self.cwd == self.dirs[-1], "Current directory is not what it is supposed to be."
            logging.info("Changed to previous directory. Current directory is {}.".format(self.cwd))

        return self.cwd

    def chdir_base(self):
        """
        Method that goes back to the base directory.

        Returns
        -------
        cwd : str
            Current working directory.
        """
        # Sanity check
        self.check_dir_exists(self.base_dir)

        os.chdir(self.base_dir)
        self.cwd = os.getcwd()

        return self.cwd

    def create_dir(self, dirs):
        """
        Method that checks if the fed directories exists and if they don't exists it creates them.

        Parameters
        ----------
        dirs : str or list of str

        Returns
        -------
        True
            Returns `True` if all directories were created without errors.
        """

        if type(dirs) is str:
            dirs = [dirs]

        for dir in dirs:
            self.check_dir_exists(dir, not_exists=True)
            os.mkdir(dir)
            # Append to dictionary of created directories
            self.created_dirs.append(os.path.abspath(dir))

        return True

    @staticmethod
    def run_subprocess(*commands, process_name=None, output_file=None, pipe=False, shell=False):
        """
        A wrapper for subprocess.check_call.

        Parameters
        ----------
        commands : str or list of str
            Shell commands to be executed. If `pipe` is True they should be passes as shell commands.
        process_name : str or None, optional
            Name of the process. If it is none, it will be equal to the first argument's word.
            Only relevant if `pipe` is `False`.
        output_file : str or None, optional
            Output name. `None` is equal to the process_name. If not `None` should include extension.
            Only relevant if `pipe` is `False`.
        pipe : bool
            Whether to use subprocess.PIPE.
        shell : bool
            Shell command to be passed to subprocess
        """
        if process_name is None:
            try:
                # Extract first word
                process_name = str(commands[0]).split()[0]
            except TypeError:
                raise AttributeError("No commands were specified.")

        if output_file is None:
            output_file = process_name + ".out"

        if pipe:
            logging.info("Run_subprocess: Regardless of input shell bool, it will be set to true as pipe is true.")
            shell=True
            piped_sproc = [subprocess.Popen(commands[0], stdout=subprocess.PIPE, shell=shell)]
            for i in range(1, len(commands)):
                sproc = subprocess.Popen(commands[i], stdin=piped_sproc[-1].stdout, stdout=subprocess.PIPE, shell=shell)
                piped_sproc.append(sproc)

            return str(piped_sproc[-1].communicate()[0], 'utf-8')
        else:
            with open(output_file, "w") as std_out, open("{}.err".format(output_file), "w") as std_err:
                try:
                    logging.info("Running process {} with commands {}.".format(process_name, commands))
                    subprocess.check_call(commands, shell=shell, stdout=std_out, stderr=std_err)
                except subprocess.CalledProcessError:
                    logging.debug("Process {} failed with commands {}".format(process_name, commands))
                    raise OSError("Process {} failed with commands {}".format(process_name, commands))

    @staticmethod
    def check_dir_exists(dirs, not_exists=False):
        """
        Method that checks if the fed directories exist.

        Parameters
        ----------
        dirs : str or list of str
            List or str with directory path names.
        not_exists : bool, optional, default=False
            If True, the method checks if the fed directories do not exist.

        Returns
        -------
        bool or list of bool or Exception:
            If not_exists is False, returns `True` if dir exists and Raises an Exception if it does not exist.
            If not_exists is True, returns `True` if dir does not exist and Raises an Exception if it exists.
        """
        if type(dirs) is str:
            dirs = [dirs]

        dirs_exist = []
        for dir in dirs:
            if not ((not not_exists) ^ os.path.isdir(dir)):
                dirs_exist.append(True)
            else:
                if not_exists:
                    logging.error("Directory {} already exists exist.".format(os.path.abspath(dir)))
                    raise IsADirectoryError("Directory {} already exists.".format(os.path.abspath(dir)))
                else:
                    logging.error("Directory does {} not exist.".format(os.path.abspath(dir)))
                    raise NotADirectoryError("Directory {} does not exist.".format(os.path.abspath(dir)))

        return dirs_exist[0] if len(dirs_exist) == 1 else dirs_exist

    @staticmethod
    def check_file_exists(files, not_exists=False):
        """
        Method that checks if the fed files exist.

        Parameters
        ----------
        files : str or list of str
            List or str with directory path names.
        not_exists : bool, optional, default=False
            If True, the method checks if the fed directories do not exists

        Returns
        -------
        bool or list of bool or Exception:
            If not_exists is False, returns `True` if file exists and Raises an Exception if it does not exist.
            If not_exists is True, returns `True` if file does not exist and Raises an Exception if it exists.
        """
        if type(files) is str:
            files = [files]

        files_exist = []
        for file in files:
            if (not not_exists) and os.path.isfile(file):
                files_exist.append(True)
            else:
                if not_exists:
                    logging.error("File {} already exists.".format(os.path.abspath(file)))
                    raise FileExistsError("File {} already exists.".format(os.path.abspath(file)))
                else:
                    logging.error("File {} does not exist.".format(os.path.abspath(file)))
                    raise FileNotFoundError("File {} does not exist.".format(os.path.abspath(file)))

        return files_exist[0] if len(files_exist) == 1 else files_exist


