/* Type-checks sketch.ino on a host compiler against stubbed board APIs.
 *   g++ -std=c++11 -Wall -Wextra -I. -Ici -o ci/compile_check ci/compile_check.cpp
 * This does not prove the firmware runs on an ESP32, but it catches the class
 * of error that is otherwise only found by flashing: typos, wrong types and
 * mismatched printf format specifiers. */
#include "arduino_stubs.h"




#include "../sketch.ino"

int main() {
  setup();
  for (int i = 0; i < 3; i++) loop();
  std::printf("sketch.ino type-checked and ran %d simulated loops\n", 3);
  return 0;
}
