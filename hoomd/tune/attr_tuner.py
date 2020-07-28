from abc import ABCMeta, abstractmethod


class _TuneDefinition(metaclass=ABCMeta):
    def __init__(self, target, domain=None):
        if domain is not None and not len(domain) == 2:
            raise ValueError("domain must be a sequence of length two.")
        self._domain = domain
        self._target = target

    def in_domain(self, value):
        if self.domain is None:
            return True
        else:
            lower_bound, upper_bound = self.domain
            return ((lower_bound is None or lower_bound <= value)
                    and (upper_bound is None or value <= upper_bound))

    def wrap_into_domain(self, value):
        if self._domain is None:
            return value
        else:
            lower_bound, upper_bound = self.domain
            if lower_bound is not None and value < lower_bound:
                return lower_bound
            elif upper_bound is not None and value > upper_bound:
                return upper_bound
            else:
                return value

    @property
    def x(self):
        return self._get_x()

    @property
    def max_x(self):
        if self.domain is None:
            return None
        else:
            return self.domain[1]

    @property
    def min_x(self):
        if self.domain is None:
            return None
        else:
            return self.domain[0]

    @x.setter
    def x(self, value):
        if self.in_domain(value):
            return self._set_x(value)
        else:
            raise ValueError("Cannot set to a value outside the domain "
                             "{}.".format(self.domain))

    @property
    def y(self):
        return self._get_y()

    @property
    def target(self):
        return self._get_target()

    @target.setter
    def target(self, value):
        self._set_target(value)

    @abstractmethod
    def _get_x(self):
        pass

    @abstractmethod
    def _set_x(self):
        pass

    @abstractmethod
    def _get_y(self):
        pass

    def _get_target(self):
        return self._target

    def _set_target(self, value):
        self._target = value

    @property
    def domain(self):
        if self._domain is not None:
            return tuple(self._domain)
        else:
            return None


class ManualTuneDefinition(metaclass=ABCMeta):
    def __init__(self, get_y, target_y, get_x, set_x, domain=None):
        self._get_y = get_y
        self._target = target_y
        self._get_x = get_x
        self._set_x = set_x
        if domain is not None and not len(domain) == 2:
            raise ValueError("domain must be a sequence of length two.")
        self._domain = domain

    def get_x(self):
        return self._get_x()

    def set_x(self, value):
        return self._set_x(value)

    def get_y(self):
        return self._get_y()

    def get_target(self):
        return self._target


class AttrTuner(metaclass=ABCMeta):
    @abstractmethod
    def _solve_one(self, tunable):
        pass

    def solve(self, tunables):
        for tunable in tunables:
            self._solve_one(tunable)

    def tuned(self, tunables, tol):
        return all(abs(t.target - t.y) <= tol for t in tunables)


class _PositiveAttrTuner(AttrTuner):
    def __init__(self, max_scale=2.0, gamma=2.0):
        self.max_scale = max_scale
        self.gamma = gamma

    def _solve_one(self, tunable):
        x, y, target = tunable.x, tunable.y, tunable.target
        if y > 0:
            scale = ((1.0 + self.gamma) * y) / (target + (self.gamma * y))
        else:
            # y was zero. Try a value an order of magnitude smaller
            scale = 0.1
        if (scale > self.max_scale):
            scale = self.max_scale
        # Ensures we stay within the tunable's domain (i.e. we don't take on
        # values to high or low).
        tunable.x = tunable.wrap_into_domain(scale * x)
