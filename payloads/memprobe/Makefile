# ts5-memprobe: make PS5_PAYLOAD_SDK=/path/to/ps5-payload-sdk
ifndef PS5_PAYLOAD_SDK
    $(error PS5_PAYLOAD_SDK is undefined)
endif
include $(PS5_PAYLOAD_SDK)/toolchain/prospero.mk

ts5-memprobe.elf: main.c
	$(CC) -Wall -Werror -O2 -o $@ $^

clean:
	rm -f ts5-memprobe.elf
