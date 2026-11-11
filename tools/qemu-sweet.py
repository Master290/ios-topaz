#!/usr/bin/env python3
"""Boot a SWEETXNU payload (tools/mkpayload.py) in QEMU's virt board.

  qemu-sweet.py PAYLOAD OUTDIR [--secs N] [--gdb]

Does what shim/loader.c does on the phone, but on the host: places the
kernel segments, ADT and boot_args at their physical addresses and starts
the CPU at a 6-instruction stub that jumps to the entry with x0 = boot_args.
virt RAM starts at 0x40000000; with 4G it covers the phone's framebuffer
(0x5c000000 on topaz, see FB_BASE) and the kernel's phys_base (0xb0000000)
as plain memory. Default CPU is cortex-a35 (ARMv8.0: PAN=0, TGran16=1).

After N seconds (or when the guest stops) it writes into OUTDIR:
  fb.png     the 1080x2400 framebuffer (what the phone screen would show)
  regs.txt   QEMU's view of CPU state
  qemu.log   guest errors (add -d int with --trace-int; very slow in userland)

With --kernel KERNEL (the matching unstripped Mach-O), also decode the
framebuffer using its iso_font into fb.txt, take mid-run snapshots
(fb-55s.png / fb-75s.png by default; --mid-fb to change or disable), and
dump the 4 MiB sweet_log_ring (address from _sweet_log_ring) at stop.
--fail-on-panic returns 2 when a panic is visible; absence of a panic
does not prove boot. An undecodable or blank screen returns 3 with
--fail-on-panic. --keep-ram leaves ram.bin in OUTDIR (deleted by default).
"""
import argparse, json, mmap, os, socket, struct, subprocess, sys, time, zlib

FB_BASE, FB_W, FB_H = 0x5c000000, 1080, 2400  # must match tools/mkadt.py FB (topaz)
FB_STRIDE = FB_W * 4
PAGE = 0x4000
STUB_PA = 0x40200000	# above the dtb QEMU places at RAM base
BOOT_LINE_LENGTH = 608
SWEET_LOG_RING_SIZE = 4 << 20  # bsd/dev/sweet_ufs.c persistent console ring


def nm_addr(symbols, name):
    addrs = [int(line.split()[0], 16) for line in symbols.splitlines()
             if line.split()[-1:] == [name]]
    return addrs[0] if len(addrs) == 1 else None


def parse_mid_fb(s):
    if s in ('', '-', 'none'):
        return []
    return [float(x) for x in s.split(',') if x.strip()]


def build_image(payload, ram_path=None):
    """Return (phys_base, ram_image_or_path, boot_args_pa, entry_pa).

    If ram_path is set, write a sparse ram.bin there and return the path
    instead of a bytes object (needed for a multi-gigabyte rd=md0).
    """
    magic, ver, nsegs = struct.unpack_from('<8sII', payload, 0)
    assert magic == b'SWEETXNU' and ver == 1, 'not a SWEETXNU v1 payload'
    virt_base, entry_va, adt_off, adt_size, kernel_end, phys_base, mem_size = \
        struct.unpack_from('<7Q', payload, 0x10)
    cmdline = payload[0x48:0x48 + BOOT_LINE_LENGTH]
    delta = phys_base - virt_base
    adt_pa = kernel_end + delta
    ba_pa = (adt_pa + adt_size + PAGE - 1) & ~(PAGE - 1)
    top = ba_pa + PAGE
    ba = struct.pack('<HH4x4Q6QI4xQI', 2, 2, virt_base, phys_base, mem_size, top,
                     FB_BASE, 1, FB_STRIDE, FB_W, FB_H, 32, 0, adt_pa - delta, adt_size)
    ba += cmdline + b'\0' * 4 + struct.pack('<QQ', 0, mem_size)
    assert len(ba) == 736
    size = top - phys_base
    if ram_path:
        with open(ram_path, 'wb') as f:
            f.truncate(size)
            for i in range(nsegs):
                vmaddr, vmsize, foff, fsize = struct.unpack_from('<4Q', payload, 0x2a8 + 32 * i)
                o = vmaddr + delta - phys_base
                f.seek(o)
                f.write(payload[foff:foff + fsize])
            f.seek(adt_pa - phys_base)
            f.write(payload[adt_off:adt_off + adt_size])
            f.seek(ba_pa - phys_base)
            f.write(ba)
        return phys_base, ram_path, ba_pa, entry_va + delta
    img = bytearray(size)
    for i in range(nsegs):
        vmaddr, vmsize, foff, fsize = struct.unpack_from('<4Q', payload, 0x2a8 + 32 * i)
        o = vmaddr + delta - phys_base
        img[o:o + fsize] = payload[foff:foff + fsize]
    img[adt_pa - phys_base:adt_pa - phys_base + adt_size] = payload[adt_off:adt_off + adt_size]
    img[ba_pa - phys_base:ba_pa - phys_base + len(ba)] = ba
    return phys_base, bytes(img), ba_pa, entry_va + delta


def stub(ba_pa, entry_pa):
    # ldr x0, =ba ; ldr x1, =entry ; br x1 ; nop ; .quad ba ; .quad entry
    return struct.pack('<4I2Q', 0x58000080, 0x580000a1, 0xd61f0020, 0xd503201f, ba_pa, entry_pa)


def write_png(path, raw):
    rows = []
    for y in range(FB_H):
        line = raw[y * FB_STRIDE:(y + 1) * FB_STRIDE]
        rgb = bytearray(FB_W * 3)	# a8r8g8b8 little endian = B G R A in memory
        rgb[0::3], rgb[1::3], rgb[2::3] = line[2::4], line[1::4], line[0::4]
        rows.append(b'\0' + bytes(rgb))
    chunk = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d))
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', FB_W, FB_H, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(b''.join(rows), 6)) + chunk(b'IEND', b'')
    open(path, 'wb').write(png)


def payload_data(payload, va, size):
    """Read initialized segment bytes by VA (the payload has no slide)."""
    nsegs = struct.unpack_from('<I', payload, 12)[0]
    for i in range(nsegs):
        vmaddr, _, foff, fsize = struct.unpack_from('<4Q', payload, 0x2a8 + 32 * i)
        if vmaddr <= va and va + size <= vmaddr + fsize:
            offset = foff + va - vmaddr
            return payload[offset:offset + size]
    raise ValueError(f'VA {va:#x}+{size:#x} not in initialized payload segments')


def decode_screen(raw, font):
    """Exact glyph matching for pe_serial.c's 2x, white-on-black fbcon."""
    glyphs = {font[c * 16:(c + 1) * 16]: chr(c) for c in range(32, 127)}
    rows, unknown = [], 0
    for row in range(FB_H // 32):
        chars = []
        for col in range(FB_W // 16):
            glyph = bytearray(16)
            for y in range(16):
                offset = (row * 32 + y * 2) * FB_STRIDE + col * 16 * 4
                for x in range(8):
                    if raw[offset + x * 8:offset + x * 8 + 4] == b'\xff' * 4:
                        glyph[y] |= 1 << x
            key = bytes(glyph)
            char = glyphs.get(key, '?')
            unknown += key not in glyphs
            chars.append(char)
        rows.append(''.join(chars).rstrip())
    return '\n'.join(rows).rstrip() + '\n', unknown


def screen_result(screen, unknown):
    if 'panic(cpu ' in screen or 'SWEET EARLY PANIC:' in screen:
        status = 'panic'
    elif unknown:
        status = 'incomplete_decode'
    elif screen.strip():
        status = 'no_panic_observed'
    else:
        status = 'blank_screen'
    return {'status': status, 'unknown_glyphs': unknown, 'snapshot_only': True}


def qmp(sock_path, *cmds, pmemsave=None):
    """Run QEMU human-monitor commands over QMP (synchronous, unlike the echoing HMP socket)."""
    s = socket.socket(socket.AF_UNIX)
    s.connect(sock_path)
    f = s.makefile('rw')
    f.readline()

    def call(execute, **args):
        f.write(json.dumps({'execute': execute, 'arguments': args}) + '\n')
        f.flush()
        while True:
            r = json.loads(f.readline())
            if 'return' in r or 'error' in r:
                return r

    call('qmp_capabilities')
    out = []
    for c in cmds:
        r = call('human-monitor-command', **{'command-line': c})
        if 'error' in r:
            print(f'qmp {c!r}: {r["error"]}', file=sys.stderr)
        out.append(r.get('return', ''))
    if pmemsave:	# native QMP: HMP's argument parser chokes on '-' in paths
        r = call('pmemsave', val=pmemsave[0], size=pmemsave[1], filename=pmemsave[2])
        if 'error' in r:
            print(f'qmp pmemsave: {r["error"]}', file=sys.stderr)
    s.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('payload'); ap.add_argument('outdir')
    ap.add_argument('--secs', type=float, default=20)
    ap.add_argument('--gdb', action='store_true', help='start halted, gdb stub on :1234, no timeout')
    ap.add_argument('--hmp', action='append', default=[], help='extra monitor command run after stop (repeatable), e.g. "x /8gx 0xfffffe..."')
    # Official SWEET kernel is compiled ARMv8.0 (0081: no LSE/PAN). Real Kryo 260 is
    # A53/A73, ID_AA64MMFR1.PAN=0. cortex-a76 is ARMv8.2 with PAN=2 and 16K pages;
    # PAN is auto-set on EL1 exceptions when SCTLR.SPAN is unset, which this kernel
    # never programs. a53/a57/a72 have PAN=0 but TGran16=0 (16K not advertised).
    ap.add_argument('--cpu', default='cortex-a35', help='QEMU CPU: a35 is ARMv8.0 with TGran16=1 and PAN=0; a53/a72 have no 16K; a76 has PAN+16K')
    ap.add_argument('--trace-int', action='store_true', help='log every guest exception to qemu.log (-d int); very slow once userspace faults')
    ap.add_argument('--kernel', help='matching unstripped kernel, for exact framebuffer text decoding (uses llvm-nm)')
    ap.add_argument('--fail-on-panic', action='store_true', help='exit 2 for a visible panic, 3 for incomplete/blank decoding; requires --kernel')
    ap.add_argument('--keep-ram', action='store_true', help='leave ram.bin in OUTDIR (deleted by default)')
    ap.add_argument('--mid-fb', default='55,75', help='comma-separated seconds for mid-run FB snapshots when --kernel is set; "-" disables')
    a = ap.parse_args()
    if a.fail_on_panic and not a.kernel:
        ap.error('--fail-on-panic requires --kernel')
    payload_f = open(a.payload, 'rb')
    payload = mmap.mmap(payload_f.fileno(), 0, access=mmap.ACCESS_READ)
    font = None
    ring_va = None
    if a.kernel:
        symbols = subprocess.check_output(['llvm-nm', '--defined-only', a.kernel], text=True)
        font_addr = nm_addr(symbols, '_iso_font')
        if font_addr is None:
            ap.error('kernel must contain exactly one _iso_font symbol')
        font = payload_data(payload, font_addr, 256 * 16)
        ring_va = nm_addr(symbols, '_sweet_log_ring')
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.abspath(a.outdir)

    phys_base, _ram, ba_pa, entry_pa = build_image(payload, ram_path=f'{out}/ram.bin')
    open(f'{out}/stub.bin', 'wb').write(stub(ba_pa, entry_pa))
    print(f'phys_base={phys_base:#x} boot_args={ba_pa:#x} entry={entry_pa:#x}', file=sys.stderr)

    mon = f'{out}/mon.sock'
    if os.path.exists(mon):
        os.unlink(mon)
    # virt RAM starts at 0x40000000: cover the payload's RAM (mkadt.py --highmem needs 8G).
    virt_base = struct.unpack_from('<Q', payload, 0x10)[0]
    mem_size = struct.unpack_from('<Q', payload, 0x40)[0]
    # Drop the 4GiB payload mmap before QEMU starts; the font is already in memory.
    payload.close()
    payload_f.close()
    ram_gib = max(4, -(-(phys_base + mem_size - 0x40000000) // (1 << 30)))
    cmd = ['qemu-system-aarch64', '-M', 'virt,gic-version=3', '-cpu', a.cpu, '-smp', '1', '-m', f'{ram_gib}G',
           '-nographic', '-serial', f'file:{out}/serial.txt', '-qmp', f'unix:{mon},server,nowait',
           '-d', ('int,guest_errors,unimp' if a.trace_int else 'guest_errors,unimp'), '-D', f'{out}/qemu.log', '-no-reboot',
           '-device', f'loader,file={out}/ram.bin,addr={phys_base:#x},force-raw=on',
           '-device', f'loader,file={out}/stub.bin,addr={STUB_PA:#x},force-raw=on',
           '-device', f'loader,addr={STUB_PA:#x},cpu-num=0']
    if a.gdb:
        cmd += ['-S', '-gdb', 'tcp::1234']
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
    if a.gdb:
        print('gdb stub on :1234; QMP socket:', mon, file=sys.stderr)
        p.wait()
        return
    t0 = time.time()
    mid_at = parse_mid_fb(a.mid_fb)
    mid_done = set()
    while p.poll() is None and time.time() - t0 < a.secs:
        time.sleep(0.2)
        elapsed = time.time() - t0
        for tsec in mid_at:
            if tsec not in mid_done and elapsed >= tsec and a.kernel and font is not None:
                raw_path = f'{out}/fb-{tsec}s.raw'
                try:
                    qmp(mon, pmemsave=(FB_BASE, FB_STRIDE * FB_H, raw_path))
                    rawm = open(raw_path, 'rb').read()
                    write_png(f'{out}/fb-{tsec}s.png', rawm)
                    screen, unknown = decode_screen(rawm, font)
                    open(f'{out}/fb-{tsec}s.txt', 'w').write(screen)
                    os.unlink(raw_path)
                    print(f'mid-snapshot {tsec}s unknown={unknown}', file=sys.stderr)
                except Exception as e:
                    print(f'mid-snapshot {tsec}s failed: {e}', file=sys.stderr)
                mid_done.add(tsec)
    result = {'status': 'qemu_exited', 'qemu_returncode': p.returncode}
    if p.poll() is None:
        # Dump the 4 MiB sweet_log_ring when --kernel resolved _sweet_log_ring.
        if ring_va is not None:
            ring_pa = phys_base + (ring_va - virt_base)
            try:
                qmp(mon, pmemsave=(ring_pa, SWEET_LOG_RING_SIZE, f'{out}/sweet_log_ring.bin'))
            except Exception as e:
                print(f'sweet_log_ring pmemsave failed: {e}', file=sys.stderr)
        _, regs, *extra = qmp(mon, 'stop', 'info registers', *a.hmp, pmemsave=(FB_BASE, FB_STRIDE * FB_H, f'{out}/fb.raw'))
        open(f'{out}/regs.txt', 'w').write(regs + ''.join(f'\n(qemu) {c}\n{r}' for c, r in zip(a.hmp, extra)))
        p.kill()
        p.wait()
        raw = open(f'{out}/fb.raw', 'rb').read()
        write_png(f'{out}/fb.png', raw)
        result = {'status': 'not_decoded'}
        if font is not None:
            screen, unknown = decode_screen(raw, font)
            open(f'{out}/fb.txt', 'w').write(screen)
            result = screen_result(screen, unknown)
        os.unlink(f'{out}/fb.raw')
    else:
        print(f'qemu exited early: {p.returncode}', file=sys.stderr)
    if not a.keep_ram:
        for f in ('ram.bin',):
            try:
                os.unlink(f'{out}/{f}')
            except FileNotFoundError:
                pass
    open(f'{out}/result.json', 'w').write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), file=sys.stderr)
    print(f'{out}/fb.png  {out}/regs.txt  {out}/qemu.log', file=sys.stderr)
    if a.fail_on_panic and result['status'] == 'panic':
        return 2
    if a.fail_on_panic and result['status'] in ('incomplete_decode', 'blank_screen'):
        return 3
    if result['status'] == 'qemu_exited':
        return 1


if __name__ == '__main__':
    sys.exit(main())
