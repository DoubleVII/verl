namespace py lab.mt.bleurt
namespace go lab.mt.bleurt
namespace rs lab.mt.bleurt

include "base.thrift"

struct BleurtReq {
    1: required list<string>            reference_list    
    2: required list<string>            candidate_list    
    4: optional map<string, string>     options                
    255: required base.Base Base
}
struct BleurtResp {
    1: optional list<double> score_list 
    255: required base.BaseResp BaseResp
}

service BleurtService {
    BleurtResp score(1: BleurtReq req)
}
