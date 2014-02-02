import numpy as np
from mako.template import Template
from mako.runtime import Context
import os.path
from subprocess import check_call
from timethis import timethis

TMP_DIR = "./"

R_script = """
library("ggplot2")

raw_data <- read.csv("${csv_file}");
names(raw_data) <- sub(" ", "_", names(raw_data));

p <- ggplot(raw_data);

p <- p + geom_point(aes(x=${x_col}, y=${y_col}, color=${w_col}));
p <- p + scale_color_gradient(low="blue", high="red");
#p <- p + scale_alpha(range=c(1.0, 0.0));
% if with_facets:
p <- p + facet_grid(${facet_1} ~ ${facet_2});
% endif

% if xlab is not None:
p <- p + xlab("${xlab}");
% endif
% if ylab is not None:
p <- p + ylab("${ylab}");
% endif
% if title is not None:
p <- p + ggtitle("${title}");
% endif

% if with_facets:
ggsave("${prefix}_param.pdf", width=20, height=12);
% else:
ggsave("${prefix}_param.pdf", width=10, height=6);
% endif


p <- ggplot(raw_data);
p <- p + geom_histogram(aes(x=${w_col}), binwidth=0.001);
% if with_facets:
p <- p + facet_grid(${facet_1} ~ ${facet_2});
% endif

% if with_facets:
ggsave("${prefix}_error.pdf", width=20, height=12);
% else:
ggsave("${prefix}_error.pdf", width=10, height=6);
% endif
""";

@timethis
def generate_R_script(filename, x_col, y_col, w_col, options):
    r_template = Template(R_script);
    r_file = os.path.join(TMP_DIR, "plot.r");

    with open(r_file, 'w') as fout:
        ctx = Context(fout,
                csv_file = filename,
                x_col = x_col,
                y_col = y_col,
                w_col = w_col,
                **options);
        r_template.render_context(ctx);
    return r_file;

@timethis
def point_plot(x, y, w, angles, ratios, xlab="x", ylab="y", wlab="weight", title=None,
        out_file=None, err_bound=None, facet_1=None, facet_2=None, prefix=""):
    assert(len(x) == len(y));
    assert(len(w) == len(x));
    assert(angles.shape[0] == ratios.shape[0]);
    assert(angles.shape[1] == len(x));
    assert(ratios.shape[1] == len(x));

    num_samples = len(x);
    num_laminates = angles.shape[0];
    data_file = os.path.join(TMP_DIR, "data.txt");
    with open(data_file, 'w') as fout:
        colume_names = "{}, {}, {}".format(xlab, ylab, wlab);
        for i in range(num_laminates):
            colume_names += ", angle_{}, ratio_{}".format(i, i);
        fout.write("{}\n".format(colume_names));

        for i, x_val, y_val, w_val in zip(range(num_samples), x, y, w):
            row = "{}, {}, {}".format(x_val, y_val, w_val);
            for j in range(num_laminates):
                row += ", {}, {}".format(angles[j,i], ratios[j,i]);
            fout.write("{}\n".format(row));

    options = {
            "prefix":prefix,
            "xlab":xlab,
            "ylab":ylab,
            "title":title,
            "err_bound":err_bound,
            "facet_1":facet_1,
            "facet_2":facet_2,
            "with_facets":(facet_1 is not None and facet_2 is not None)
            };
    r_file = generate_R_script(data_file, xlab, ylab, wlab, options);

    command = "Rscript {}".format(r_file);
    check_call(command.split());

