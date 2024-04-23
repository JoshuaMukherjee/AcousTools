if __name__ == '__main__':
    from acoustools.Utilities import create_points, add_lev_sig, write_to_file
    from acoustools.Solvers import wgs


    p = create_points(1,1,x=0,y=0,z=0)
    x = wgs(p)
    x = add_lev_sig(x)

    write_to_file(x,'wgsTrap.csv',1)



