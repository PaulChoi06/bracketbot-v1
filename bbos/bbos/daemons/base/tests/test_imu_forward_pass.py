import ctypes as C
import importlib.util
import math
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

BASE=Path(__file__).resolve().parents[1]
FW=BASE/'firmware/stm/balance'
ORIGINAL=BASE.parents[1]/'bracketbot-188-original/base/firmware/stm/balance'

class ImuForwardTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.d=Path(cls.tmp.name)
  stub='''#include <stdint.h>
#include <stddef.h>
typedef struct { uint32_t CYCCNT; } TestDWT;
extern TestDWT test_dwt;
#define DWT (&test_dwt)
typedef struct { int dummy; } SPI_HandleTypeDef;
extern SPI_HandleTypeDef hspi6;
typedef int HAL_StatusTypeDef;
#define HAL_OK 0
#define GPIOA ((void*)0)
#define GPIO_PIN_0 1
#define GPIO_PIN_RESET 0
#define GPIO_PIN_SET 1
void HAL_GPIO_WritePin(void*,int,int);
int HAL_SPI_TransmitReceive(SPI_HandleTypeDef*,uint8_t*,uint8_t*,uint16_t,uint32_t);
int HAL_SPI_Transmit(SPI_HandleTypeDef*,uint8_t*,uint16_t,uint32_t);
void HAL_Delay(uint32_t);
uint32_t HAL_GetTick(void);
'''
  (cls.d/'board.h').write_text(stub)
  harness='''#include "board.h"
#include "imu_runtime.h"
#include "imu_diag.h"
#include "imu_filter.h"
#include <string.h>
TestDWT test_dwt;
SPI_HandleTypeDef hspi6;
static int ready,fail;
static uint8_t config0;
static float dt_seen;
void HAL_GPIO_WritePin(void*a,int b,int c){(void)a;(void)b;(void)c;}
int HAL_SPI_TransmitReceive(SPI_HandleTypeDef*h,uint8_t*t,uint8_t*r,uint16_t n,uint32_t x){
 (void)h;(void)x;memset(r,0,n);test_dwt.CYCCNT+=500;
 if(fail)return 1;
 unsigned reg=t[0]&127;
 if(reg==0x75)r[1]=0x47;
 if(reg==0x2d)r[1]=ready?8:0;
 if(reg==0x1d){r[5]=0x20;ready=0;}
 return 0;
}
int HAL_SPI_Transmit(SPI_HandleTypeDef*h,uint8_t*t,uint16_t n,uint32_t x){
 (void)h;(void)n;(void)x;if(t[0]==0x63)config0=t[1];return 0;
}
void HAL_Delay(uint32_t x){test_dwt.CYCCNT+=x*250000u;}
uint32_t HAL_GetTick(void){return test_dwt.CYCCNT/250000u;}
void link_send_imu_diagnostics(const uint32_t*w){(void)w;}
int real_imu_filter_update(imu_filter_t*,const imu_sample_t*,float,float);
int imu_filter_update(imu_filter_t*f,const imu_sample_t*s,float dt,float beta){dt_seen=dt;return real_imu_filter_update(f,s,dt,beta);}
void policy_background_poll(void);
void test_init(uint32_t clock){
 test_dwt.CYCCNT=clock;ready=fail=0;dt_seen=0;
 g_imu_status=g_imu_sample_count=g_imu_last_sample_cycle=0;
 memset(g_imu_diag,0,4*IMU_DIAG_COUNT);g_imu_diag[IMU_DIAG_DT_MIN_CYCLES]=UINT32_MAX;
 imu_runtime_init();float b[3]={0};imu_runtime_set_gyro_bias(b);
}
void test_poll(uint32_t delta,int r,int f,int forward){test_dwt.CYCCNT+=delta;ready=r;fail=f;if(forward)policy_background_poll();else imu_runtime_poll();}
uint32_t test_diag(int i){return g_imu_diag[i];}
float test_dt(void){return dt_seen;}
int test_config0(void){return config0;}
'''
  (cls.d/'harness.c').write_text(harness)
  common=['cc','-O3','-fPIC','-fno-math-errno','-ffp-contract=fast','-I'+str(cls.d),'-I'+str(FW)]
  subprocess.run(common+['-Dimu_filter_update=real_imu_filter_update','-c',str(FW/'imu_filter.c'),'-o',str(cls.d/'filter.o')],check=True)
  subprocess.run(common+['-shared',str(FW/'baseboard_imu_runtime.c'),str(FW/'imu_icm42688.c'),str(cls.d/'harness.c'),str(cls.d/'filter.o'),'-lm','-o',str(cls.d/'imu.so')],check=True)
  cls.lib=C.CDLL(str(cls.d/'imu.so'));cls.lib.test_init.argtypes=[C.c_uint32]
  cls.lib.test_poll.argtypes=[C.c_uint32,C.c_int,C.c_int,C.c_int]
  cls.lib.test_diag.restype=C.c_uint32;cls.lib.test_dt.restype=C.c_float
  spec=importlib.util.spec_from_file_location('driver188',BASE/'driver.py');cls.driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.driver)
  cls.fields=cls.driver.IMU_DIAG_FIELDS
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def setUp(self):self.lib.test_init(0)
 def diag(self,key):return self.lib.test_diag(self.fields.index(key))
 def test_no_integration_without_fresh_sensor_data(self):
  self.lib.test_poll(125000,0,0,0)
  self.assertEqual(self.diag('not_ready'),1);self.assertEqual(self.diag('filter_updates'),0)
  self.assertEqual(self.lib.test_config0(),0x20)
 def test_forward_pass_callback_acquires_and_filters(self):
  self.lib.test_poll(125000,1,0,1)
  self.assertEqual(self.diag('forward_samples'),1);self.assertEqual(self.diag('filter_updates'),1)
  self.lib.test_poll(125000,0,0,1)
  self.assertEqual(self.diag('forward_samples'),1)
 def test_actual_interval_survives_long_gap(self):
  self.lib.test_poll(125000,1,0,0);self.lib.test_poll(6250000,1,0,0)
  self.assertAlmostEqual(self.lib.test_dt(),(6250000+1000)/250000000,places=7)
  self.assertEqual(self.diag('filter_updates'),2);self.assertGreater(self.diag('estimated_missing'),0)
 def test_cycle_wrap_preserves_interval(self):
  self.lib.test_init(0xffffffff-55000000)
  for _ in range(40):self.lib.test_poll(125000,1,0,0)
  self.assertEqual(self.diag('filter_updates'),40)
  self.assertAlmostEqual(self.lib.test_dt(),126000/250000000,places=7)
 def test_bus_failure_is_counted_without_filtering(self):
  self.lib.test_poll(125000,1,1,0)
  self.assertEqual(self.diag('bus_errors'),1);self.assertEqual(self.diag('filter_updates'),0)
 def test_diagnostics_fragmentation_and_crc_recovery(self):
  dr=self.driver;link=dr.Link.__new__(dr.Link)
  link.fd=0;link.buf=bytearray();link._can_buf=bytearray();link.expected_version=9;link.crc_errors=0;link.imu_diag=None
  values=tuple(range(24));body=struct.pack('<HBBI24I',dr.STATUS_SYNC,9,7,24,*values)
  packet=body+struct.pack('<H',dr.crc16_ccitt(body))
  bad=bytearray(packet);bad[-1]^=1
  status=bytearray(142);struct.pack_into('<HBB',status,0,dr.STATUS_SYNC,9,2);struct.pack_into('<H',status,140,dr.crc16_ccitt(status[:-2]))
  decoded=[]
  with patch.object(dr.select,'select',return_value=([],[],[])):
   for byte in bytes(bad)+packet+status:
    link.buf.append(byte);decoded.extend(link.read(0))
  self.assertEqual(link.imu_diag,values);self.assertEqual(link.crc_errors,1);self.assertEqual(len(decoded),1)
 def test_policy_math_and_both_modes_unchanged(self):
  if not ORIGINAL.exists():self.skipTest('Original source snapshot is only available in the deployment workspace')
  hook=self.d/'hook.c';hook.write_text('unsigned calls; void policy_background_poll(void){calls++;}')
  libs=[]
  for name,path in [('original',ORIGINAL),('patched',FW)]:
   output=self.d/(name+'.so')
   subprocess.run(['cc','-shared','-fPIC','-O3','-ffp-contract=fast',str(path/'policy.c'),str(hook),'-lm','-o',str(output)],check=True)
   lib=C.CDLL(str(output));lib.policy_init();libs.append(lib)
  rng=random.Random(188)
  for mode,n in [('policy_forward',18),('policy_forward_lean',19)]:
   for _ in range(1000):
    obs=(C.c_float*n)(*(rng.uniform(-3,3) for _ in range(n)));a=(C.c_float*2)();b=(C.c_float*2)()
    getattr(libs[0],mode)(obs,a);getattr(libs[1],mode)(obs,b)
    self.assertEqual(bytes(a),bytes(b))
  self.assertGreater(C.c_uint.in_dll(libs[1],'calls').value,1000)
  self.assertEqual(C.c_uint.in_dll(libs[0],'calls').value,0)

if __name__=='__main__':unittest.main(verbosity=2)
