from typing import List, MutableMapping
import numpy as np

UserID = str
InputData = List[str]
MetaData = List[str]
UserVectorData = MutableMapping[UserID, np.ndarray]
UserFeatureMap = MutableMapping[UserID, int]
UserMetaData = MutableMapping[UserID, MetaData]
HashResult = MutableMapping[UserID, int]
