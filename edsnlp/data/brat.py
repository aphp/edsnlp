from edsnlp.data.standoff import (
    dump_standoff_file,
    parse_standoff_file,
    read_standoff,
    write_standoff,
)

load_from_brat = parse_standoff_file
export_to_brat = dump_standoff_file

read_brat = read_standoff
write_brat = write_standoff
